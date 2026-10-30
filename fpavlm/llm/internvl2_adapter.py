"""InternVL2-2B LLM Adapter — full plug-and-play implementation (v170 FIX).

WHY InternVL2-2B:
  - small VLM (2B params, InternLM2.5-1.8B backbone + InternViT-300M)
  - 4-bit NF4 quantization → ~1.5GB VRAM (fits in 4070 8GB alongside FHE pipeline)
  - Strong general VLM performance (MMBench 73.5, MME 1755)
  - Active maintenance (HuggingFace + OpenGVLab)
  - Plug-and-play compatible with FHEViT framework (image token injection via
    native `visual_features` parameter — NO manual masked_scatter needed)

PLUG-AND-PLAY CONTRACT:
  - Input: [1, N_img, llm_hidden_dim] = [1, 256, 2048] from MedicalConceptAdapter
    (Perceiver expands 16 → 256 tokens, llm_hidden_dim=2048 for InternVL2-2B)
  - Output: medical report text
  - Image embedding injection: via InternVL2's NATIVE `visual_features` parameter
    in `model.generate()`. InternVL2's generate() does:
      vit_embeds = visual_features  (our pre-computed features)
      input_embeds[selected] = vit_embeds  (native injection at IMG_CONTEXT)
    This is the CORRECT injection path — no manual masked_scatter needed.
  - Chat template: InternVL2's native internvl2_5 template WITH system message
    (v170 FIX: was missing system message + missing \n after </img>, causing
    garbage output "a one a one a name the idea!")

v170 FIX (FIRST-PRINCIPLES DIAGNOSIS):
  ROOT CAUSE of garbage output in v167-v169:
    1. The adapter BYPASSED `model.generate()` and called
       `GenerationMixin.generate(language_model)` directly. This skipped
       InternVL2's native `visual_features` injection path.
    2. The chat template was MISSING the system message
       (`<|im_start|>system\n你是书生·万象...<|im_end|>`). The model was
       trained with this system message — without it, the model doesn't
       recognize the format and produces incoherent output.
    3. The chat template was MISSING the `\n` after `</img>` before the prompt.
       InternVL2's `chat()` method formats as `<image>\n{question}`, which
       becomes `<img>...<IMG_CONTEXT>×N...</img>\n{question}`. Without the
       `\n`, the image section and prompt are concatenated, confusing the model.

  FIX:
    1. Use `model.generate(visual_features=image_embeds, pixel_values=dummy,
       input_ids=correct_ids)` — the NATIVE generate path. InternVL2's
       generate() handles feature injection, KV cache, and generation config.
    2. Include the system message in the chat template.
    3. Add `\n` after `</img>` before the prompt.
    4. Keep the DynamicCache patches (for transformers 4.50+ compat).

v167 INTEGRATION:
  - Registered as "internvl2-2b" in LLMRegistry
  - FHEConfig: perceiver_name="v162-default" works (PerceiverDecoder llm_dim=2048)
  - 4-bit quantization via BitsAndBytesConfig (NF4, double quant, bfloat16 compute)
  - Loads from LOCAL path only (no HuggingFace download at runtime)

USAGE:
  ```python
  from fpavlm import FHEConfig, FHEViTPipeline

  config = FHEConfig(llm_hidden_dim=2048)  # InternVL2-2B hidden dim
  pipeline = FHEViTPipeline.from_pretrained(
      llm_name="internvl2-2b",
      llm_model_path="models/internvl2-2b",
      encoder_weights="artifacts/encoder_weights.pt",
      medical_priors="artifacts/medical_priors.pt",
      cfp_sensitivity="artifacts/cfp_sensitivity.pt",
      config=config,
  )
  report = pipeline.run("data/openi_subset/images/openi_00001.png")
  ```

NOTE ON EMBEDDING ADAPTATION:
  The shared encoder emits 2560-dimensional visual tokens. The required trained
  adapter at artifacts/adapters/internvl2-2b.pt maps them into InternVL2's
  2048-dimensional native visual space; zero-shot PCA is ablation-only.

DOWNLOAD INSTRUCTIONS (user must pre-download to models/internvl2-2b/):
  Option 1 (HuggingFace): huggingface-cli download OpenGVLab/InternVL2_5-2B \\
      --local-dir models/internvl2-2b --local-dir-use-symlinks False
  Option 2 (ModelScope): modelscope download --model OpenGVLab/InternVL2_5-2B \\
      --local_dir models/internvl2-2b

REFERENCES:
  - InternVL2 paper: Chen et al. "InternVL: Scaling up Vision Foundation Models
    and Aligning for Generic Visual-Linguistic Tasks" (CVPR 2024)
  - HF docs: https://huggingface.co/OpenGVLab/InternVL2_5-2B
"""
from __future__ import annotations

from typing import List, Optional

import torch

from fpavlm.llm.interface import LLMInterface, LLMRegistry

# Default medical prompt (overridable via generate(prompt=...))
DEFAULT_MEDICAL_PROMPT_INTERNVL = (
    "You are a radiologist examining a chest X-ray. "
    "Write a brief radiology report. "
    "Describe the lungs (clear or abnormal), heart size, mediastinum, "
    "diaphragm, pleural spaces, and any pathology such as "
    "consolidation, effusion, or pneumothorax."
)

# v170 FIX: InternVL2's system message (from conversation.py internvl2_5 template).
# The model was TRAINED with this system message. Without it, the model doesn't
# recognize the chat format and produces garbage output.
_INTERNVL2_SYSTEM_MESSAGE = (
    "你是书生·万象，英文名是InternVL，是由上海人工智能实验室、清华大学及多家合作单位"
    "联合开发的多模态大语言模型。"
)


def _trim_repeated_report(text: str) -> str:
    """Drop InternVL2's common self-echo after the first findings paragraph."""
    if not text:
        return text
    for stop in ("<|im_end|>", "<|im_start|>"):
        if stop in text:
            text = text.split(stop, 1)[0]
    text = text.strip()
    marker = "Radiology findings:"
    first = text.find(marker)
    if first >= 0:
        second = text.find(marker, first + len(marker))
        if second > 0:
            text = text[:second].strip()
    return text.strip()


def patch_internvl2_for_transformers_4_50(model) -> None:
    """Patch InternVL2/InternLM2 for transformers 4.50+ compatibility.

    ROOT CAUSE (v174 diagnostic):
      InternLM2's custom `prepare_inputs_for_generation` and `forward` methods
      (in modeling_internlm2.py) assume `past_key_values` is a tuple-of-tuples
      (legacy format). In transformers 4.50+, `past_key_values` is a
      `DynamicCache` object. The code:
          past_length = past_key_values[0][0].shape[2]
      returns None when indexing a DynamicCache with tuple syntax, causing:
          AttributeError: 'NoneType' object has no attribute 'shape'

    This function applies three patches (mirroring what the adapter's load()
    does internally):
      1. Add GenerationMixin to InternLM2ForCausalLM's bases (transformers
         4.50+ removed it from PreTrainedModel).
      2. Replace prepare_inputs_for_generation with a version that handles
         both DynamicCache and tuple formats.
      3. Wrap InternLM2Model.forward to convert empty DynamicCache to None
         (avoids the shape access entirely on the first forward pass).

    Args:
        model: An InternVL2 model (loaded via AutoModel with trust_remote_code)
            — must have `.language_model` and `.language_model.model` attributes.
    """
    import types

    from transformers import GenerationConfig, GenerationMixin

    # Patch 1: GenerationMixin (transformers 4.50+ removed it from PreTrainedModel)
    lm_cls = type(model.language_model)
    if GenerationMixin not in lm_cls.__mro__:
        lm_cls.__bases__ = lm_cls.__bases__ + (GenerationMixin,)
    if getattr(model.language_model, 'generation_config', None) is None:
        model.language_model.generation_config = GenerationConfig.from_model_config(
            model.language_model.config
        )

    # Patch 2: prepare_inputs_for_generation (handles DynamicCache vs tuple)
    lm = model.language_model
    if not getattr(lm, '_prepare_inputs_patched', False):
        def patched_prepare_inputs_for_generation(
            self, input_ids, past_key_values=None, attention_mask=None,
            inputs_embeds=None, **kwargs,
        ):
            if past_key_values is not None:
                if hasattr(past_key_values, 'get_seq_length'):
                    past_length = past_key_values.get_seq_length()
                elif isinstance(past_key_values, tuple) and len(past_key_values) > 0:
                    try:
                        past_length = past_key_values[0][0].shape[2]
                    except (AttributeError, IndexError, TypeError):
                        past_length = 0
                else:
                    past_length = 0

                if past_length > 0 and input_ids is not None:
                    if input_ids.shape[1] > past_length:
                        remove_prefix_length = past_length
                    else:
                        remove_prefix_length = input_ids.shape[1] - 1
                    input_ids = input_ids[:, remove_prefix_length:]

            position_ids = kwargs.get('position_ids', None)
            if attention_mask is not None and position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids = position_ids.masked_fill(attention_mask == 0, 1)
                if past_key_values is not None and input_ids is not None:
                    position_ids = position_ids[:, -input_ids.shape[1]:]

            cache_empty = (
                past_key_values is None or
                (hasattr(past_key_values, 'get_seq_length') and
                 past_key_values.get_seq_length() == 0)
            )
            if inputs_embeds is not None and cache_empty:
                model_inputs = {'inputs_embeds': inputs_embeds}
            else:
                model_inputs = {'input_ids': input_ids}

            model_inputs['position_ids'] = position_ids
            if attention_mask is not None:
                model_inputs['attention_mask'] = attention_mask
            if past_key_values is not None:
                model_inputs['past_key_values'] = past_key_values
            return model_inputs

        lm.prepare_inputs_for_generation = types.MethodType(
            patched_prepare_inputs_for_generation, lm
        )
        lm._prepare_inputs_patched = True

    # Patch 3: InternLM2Model.forward (convert empty DynamicCache to None)
    inner_model = lm.model
    if not getattr(inner_model, '_forward_patched', False):
        orig_forward = inner_model.forward
        def patched_forward(
            self, input_ids=None, attention_mask=None, position_ids=None,
            past_key_values=None, inputs_embeds=None, **kwargs,
        ):
            if past_key_values is not None:
                if hasattr(past_key_values, 'get_seq_length'):
                    if past_key_values.get_seq_length() == 0:
                        past_key_values = None
            if past_key_values is not None and not hasattr(past_key_values, 'get_seq_length'):
                try:
                    _ = past_key_values[0][0].shape[2]
                except (AttributeError, IndexError, TypeError):
                    past_key_values = None
            return orig_forward(
                input_ids=input_ids, attention_mask=attention_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                inputs_embeds=inputs_embeds, **kwargs,
            )
        inner_model.forward = types.MethodType(patched_forward, inner_model)
        inner_model._forward_patched = True


@LLMRegistry.register("internvl2-2b")
class InternVL2Adapter(LLMInterface):
    """InternVL2-2B LLM adapter (4-bit NF4 quantization for 4070 8GB).

    Implements LLMInterface for InternVL2.5-2B. Loads from local path with
    4-bit quantization. Image embeddings are injected via InternVL2's NATIVE
    `visual_features` parameter in `model.generate()` — no manual masked_scatter.

    v170 FIX: Uses `model.generate(visual_features=...)` instead of bypassing
    with `GenerationMixin.generate(language_model)`. This ensures correct
    feature injection and chat template handling.
    """

    def __init__(
        self,
        model_path: str,
        quantization: str = "4bit",
        device: str = "auto",
        **kwargs,
    ):
        super().__init__(model_path=model_path, **kwargs)
        self.quantization = quantization
        self.device = device
        self._llm_hidden_dim = 2048  # InternVL2-2B hidden dim (InternLM2.5-1.8B)
        self._image_token_id = None
        # InternVL2 uses <IMG_CONTEXT> (id=92546) as the image EMBEDDING placeholder.
        # The chat template wraps image tokens in <img>...</img> tags:
        #   <img><IMG_CONTEXT>×N</img>
        # Image embeddings are injected at <IMG_CONTEXT> positions (not <img>).
        # <img> (92544) = image section start, </img> (92545) = image section end,
        # <IMG_CONTEXT> (92546) = actual embedding placeholder.
        self._image_token_str = "<IMG_CONTEXT>"
        self._img_start_str = "<img>"
        self._img_end_str = "</img>"
        self._loaded = False

    def load(self) -> None:
        """Load InternVL2.5-2B in 4-bit NF4 quantization.

        Uses BitsAndBytesConfig for 4-bit quantization:
          - load_in_4bit=True
          - bnb_4bit_quant_type="nf4"
          - bnb_4bit_compute_dtype=torch.bfloat16
          - bnb_4bit_use_double_quant=True

        InternVL2 uses AutoModel.from_pretrained (not AutoModelForCausalLM)
        because it has a custom vision encoder + LLM fusion architecture.
        """
        from transformers import (
            AutoModel,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        self.verify_remote_code_trust()
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # InternVL2 loads via AutoModel (custom InternVLChatModel class)
        # v174 FIX: Use device_map={"": "cuda:0"} instead of "auto" to force
        # ALL layers (including embedding + language model) on GPU. With
        # device_map="auto", the embedding layer may be placed on CPU, causing
        # "_native_layer_norm" device mismatch during generate().
        # InternVL2-2B 4-bit NF4 is ~1.5GB — fits easily on 4070 8GB.
        self.model = AutoModel.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            device_map={"": "cuda:0"},
        )
        self.model.eval()

        # Load tokenizer (InternVL2 uses InternLM2 tokenizer)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, use_fast=False
        )

        # Cache image token info
        # InternVL2 uses <IMG_CONTEXT> as the image EMBEDDING placeholder.
        # The chat template is: <img><IMG_CONTEXT>×N</img>
        self._image_token_id = self.tokenizer.convert_tokens_to_ids(self._image_token_str)
        if self._image_token_id == self.tokenizer.unk_token_id:
            # Fallback: try alternative image tokens
            for tok_str in ["<IMG_CONTEXT>", "<img>", "<image>"]:
                tid = self.tokenizer.convert_tokens_to_ids(tok_str)
                if tid != self.tokenizer.unk_token_id:
                    self._image_token_id = tid
                    self._image_token_str = tok_str
                    break

        # InternVL2's generate() requires self.img_context_token_id to be set.
        # This is the <IMG_CONTEXT> token ID used for image embedding injection.
        self.model.img_context_token_id = self._image_token_id

        # Apply transformers 4.50+ compatibility patches (GenerationMixin +
        # prepare_inputs_for_generation + InternLM2Model.forward).
         # Use the same compatibility patch for adapter and native-tower execution.
        patch_internvl2_for_transformers_4_50(self.model)

        # Verify embedding layer dtype
        embed_dtype = self.model.language_model.get_input_embeddings().weight.dtype
        assert embed_dtype == torch.bfloat16, (
            f"Embedding layer dtype={embed_dtype}, expected bfloat16. "
            f"4-bit LLM inputs_embeds path requires bf16 embeddings."
        )

        self._loaded = True

    @property
    def llm_hidden_dim(self) -> int:
        return self._llm_hidden_dim

    def image_token_id(self) -> int:
        if not self._loaded:
            raise RuntimeError("InternVL2Adapter not loaded. Call load() first.")
        return self._image_token_id

    def expected_image_token_count(self) -> int:
        """InternVL2 default image token count (variable, depends on resolution).

        Default 256 (matches our PerceiverDecoder 16→256 expansion).
        InternVL2 supports variable image token counts via <image> repetition.
        """
        return 256

    def get_native_feature_stats(self) -> Optional[torch.Tensor]:
        """Get native vision feature statistics for adapter calibration.

        v170: Runs `extract_feature` on a dummy image to obtain the native
        vision feature distribution (after mlp1 projector). The dummy image
        is sufficient because we only need the SCALE (mean/std per dim), not
        the semantic content — the projector's output scale is approximately
        input-invariant due to the LayerNorm at its input.

        Returns:
            [1, N_tokens, llm_hidden_dim] native vision features, or None if
            extraction fails.
        """
        if not self._loaded:
            return None
        try:
            with torch.no_grad():
                # InternVL2 expects [B, 3, 448, 448] pixel_values
                # Use a small random image — we only need the output scale
                dummy_image = torch.randn(
                    1, 3, 448, 448, device="cuda", dtype=torch.bfloat16
                ) * 0.5
                native_features = self.model.extract_feature(dummy_image)
                return native_features  # [1, 256, 2048]
        except Exception as e:
            print(f"WARNING: get_native_feature_stats failed: {e}")
            return None

    def extract_native_visual_features(self, image_path: str) -> Optional[torch.Tensor]:
        """Run InternViT + mlp1 on a real chest X-ray.

        InternVL2's extract_feature expects ImageNet-normalized 448×448 RGB.
        The returned tokens are the native visual upper bound for this LLM.
        """
        if not self._loaded:
            return None
        import numpy as np
        from PIL import Image

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        with Image.open(image_path) as image:
            array = np.asarray(
                image.convert("RGB").resize((448, 448), Image.Resampling.BILINEAR),
                dtype=np.float32,
            ) / 255.0
        array = (array - mean) / std
        pixels = torch.from_numpy(array.copy()).permute(2, 0, 1).unsqueeze(0)
        try:
            with torch.no_grad():
                pixels = pixels.to("cuda", dtype=torch.bfloat16)
                return self.model.extract_feature(pixels)
        except Exception as exc:
            print(f"WARNING: extract_native_visual_features failed: {exc}")
            return None

    def _build_input_ids(
        self,
        image_embeds: torch.Tensor,
        prompt: str,
    ) -> tuple:
        """Build (input_ids, attention_mask) for InternVL2's native generate().

        v170 FIX: Uses the CORRECT internvl2_5 chat template format:
          <|im_start|>system\n{system_message}<|im_end|>\n
          <|im_start|>user\n<img><IMG_CONTEXT>×N</img>\n{prompt}<|im_end|>\n
          <|im_start|>assistant\n

        Key fixes vs v167-v169:
          1. INCLUDES system message (model was trained with it — without it,
             produces garbage output)
          2. ADDS \n after </img> before prompt (matches chat() method format:
             question = '<image>\n' + question, then <image> → <img>...</img>)

        Args:
            image_embeds: [1, N_img, 2048] adapter output (B=1 only).
            prompt: Text prompt for generation.

        Returns:
            (input_ids, attention_mask) on CUDA. Image embedding injection is
            handled by InternVL2's native generate(visual_features=...) — we
            do NOT do masked_scatter here.
        """
        if not self._loaded:
            raise RuntimeError("InternVL2Adapter not loaded. Call load() first.")

        if image_embeds.dim() != 3 or image_embeds.shape[0] != 1:
            raise ValueError(
                f"image_embeds must be [1, N_img, {self._llm_hidden_dim}] (B=1 only), "
                f"got shape {list(image_embeds.shape)}"
            )

        n_img_tokens = image_embeds.shape[1]

        # v170 FIX: Build CORRECT internvl2_5 chat template WITH system message
        # and \n after </img>. This matches InternVL2's chat() method output.
        image_seq = self._img_start_str + "".join([self._image_token_str] * n_img_tokens) + self._img_end_str
        text = (
            f"<|im_start|>system\n{_INTERNVL2_SYSTEM_MESSAGE}<|im_end|>\n"
            f"<|im_start|>user\n{image_seq}\n{prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        # Tokenize
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = enc["input_ids"].to("cuda")
        attention_mask = enc["attention_mask"].to("cuda")

        # Verify image token count matches what we're providing
        actual_n_img = (input_ids == self._image_token_id).sum().item()
        if actual_n_img != n_img_tokens:
            raise RuntimeError(
                f"Image token count mismatch: input_ids has {actual_n_img} "
                f"<IMG_CONTEXT> tokens, but image_embeds has {n_img_tokens} tokens. "
                f"This should not happen with correct tokenization."
            )

        return input_ids, attention_mask

    def embed_image_tokens(self, image_embeds: torch.Tensor, prompt: Optional[str] = None) -> torch.Tensor:
        """Convert adapter output to LLM inputs_embeds format.

        Args:
            image_embeds: [1, N_img, 2048] adapter output (B=1 only).
            prompt: Optional text prompt (defaults to DEFAULT_MEDICAL_PROMPT_INTERNVL).

        Returns:
            inputs_embeds: [1, seq_len, 2048] bfloat16 tensor on CUDA.
        """
        prompt = prompt if prompt is not None else DEFAULT_MEDICAL_PROMPT_INTERNVL
        # For embed_image_tokens, we build inputs_embeds manually (no generation).
        # This is used by the pipeline for feature injection without generation.
        input_ids, attention_mask = self._build_input_ids(image_embeds, prompt)
        # Build inputs_embeds with image embeddings injected (manual path for
        # embed_image_tokens — generate() uses the native visual_features path)
        image_embeds = image_embeds.to("cuda").to(torch.bfloat16)
        inputs_embeds = self.model.language_model.get_input_embeddings()(input_ids).to(torch.bfloat16)
        special_image_mask = (
            (input_ids == self._image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
        )
        embeds_flat = image_embeds.reshape(-1).contiguous().to(torch.bfloat16)
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, embeds_flat)
        return inputs_embeds

    def generate(
        self,
        image_embeds: torch.Tensor,
        prompt: str = DEFAULT_MEDICAL_PROMPT_INTERNVL,
        max_new_tokens: int = 200,
        do_sample: bool = False,
        num_beams: int = 1,
        seed: Optional[int] = 42,
        logits_processor: Optional[List] = None,
        num_return_sequences: int = 1,
        min_new_tokens: int = 24,
    ) -> str:
        """Generate medical report from image embeddings + text prompt.

        v170 FIX: Uses InternVL2's NATIVE `model.generate(visual_features=...)`
        instead of bypassing with `GenerationMixin.generate(language_model)`.

        InternVL2's generate() method has a `visual_features` parameter that
        accepts pre-computed image features. When `pixel_values` is non-None
        AND `visual_features` is provided, InternVL2 skips the vision tower
        and uses the provided features directly:
          vit_embeds = visual_features
          input_embeds[selected] = vit_embeds  (native injection)

        This is the CORRECT injection path — no manual masked_scatter needed.

        Args:
            image_embeds: [1, N_img, 2048] adapter output.
            prompt: Text prompt.
            max_new_tokens: Maximum generation length.
            do_sample: If True, use sampling; if False, greedy/beam.
            num_beams: Number of beams (1=greedy).
            seed: Random seed for reproducibility.
            logits_processor: Optional list of LogitsProcessor objects.
            num_return_sequences: Number of sequences to generate.

        Returns:
            Generated text string (medical report).
        """
        if not self._loaded:
            raise RuntimeError("InternVL2Adapter not loaded. Call load() first.")

        # v174 FIX: Handle prompt=None from pipeline.run() (which defaults to None
        # meaning "use LLM default"). Without this, None overrides the default
        # parameter and _build_input_ids receives None, producing generic output
        # instead of a radiology report.
        prompt = prompt if prompt is not None else DEFAULT_MEDICAL_PROMPT_INTERNVL

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        # Ensure image_embeds is on CUDA + bfloat16
        if image_embeds.device.type != "cuda":
            image_embeds = image_embeds.to("cuda")
        if image_embeds.dtype != torch.bfloat16:
            image_embeds = image_embeds.to(torch.bfloat16)

        # v170 FIX: Build correct input_ids (with system message + \n after </img>)
        input_ids, attention_mask = self._build_input_ids(image_embeds, prompt)
        eos_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if eos_id is None or eos_id == self.tokenizer.unk_token_id:
            eos_id = self.tokenizer.eos_token_id

        # v170 FIX: Use InternVL2's NATIVE generate(visual_features=...) path.
        # InternVL2's generate() handles:
        #   1. vit_embeds = visual_features (our pre-computed features)
        #   2. input_embeds = language_model.get_input_embeddings()(input_ids)
        #   3. input_embeds[selected] = vit_embeds (native injection at IMG_CONTEXT)
        #   4. language_model.generate(inputs_embeds=input_embeds, use_cache=True)
        #
        # pixel_values MUST be non-None to trigger the visual_features injection
        # path. The actual value is ignored when visual_features is provided.
        dummy_pixel_values = torch.zeros(1, device="cuda", dtype=torch.bfloat16)

        # NOTE: InternVL2's generate() hardcodes use_cache=True in its call to
        # language_model.generate(). We must NOT pass use_cache here (would
        # cause "got multiple values for keyword argument 'use_cache'" error).
        # internvl2_5 chat() stops on <|im_end|>, not tokenizer.eos_token_id.
        # Without that stop, beam search on slightly OOD visual tokens collapses
        # to a one-word loop ("Radiographs.").
        gen_kwargs = dict(
            pixel_values=dummy_pixel_values,
            visual_features=image_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            early_stopping=num_beams > 1,
            repetition_penalty=1.15,
            no_repeat_ngram_size=6,
            temperature=0.7 if do_sample else 1.0,
            top_p=0.9 if do_sample else 1.0,
            pad_token_id=self.tokenizer.pad_token_id or eos_id,
            eos_token_id=eos_id,
        )
        if min_new_tokens > 0:
            gen_kwargs["min_new_tokens"] = min_new_tokens
        if logits_processor is not None and len(logits_processor) > 0:
            gen_kwargs["logits_processor"] = logits_processor
        if num_return_sequences > 1:
            gen_kwargs["num_return_sequences"] = num_return_sequences
            gen_kwargs["num_beams"] = max(num_beams, num_return_sequences)

        with torch.no_grad():
            output = self.model.generate(**gen_kwargs)

        # When generate() is called with inputs_embeds (via visual_features),
        # HuggingFace returns ONLY the newly-generated tokens — no input prefix.
        generated_ids = output[0, :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return _trim_repeated_report(text)


__all__ = ["InternVL2Adapter"]
