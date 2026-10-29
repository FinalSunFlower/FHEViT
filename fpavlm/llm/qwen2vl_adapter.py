"""Qwen2-VL-2B LLM Adapter — full plug-and-play implementation (v167).

WHY Qwen2-VL-2B:
  - SOTA small VLM (2B params, Qwen2-1.5B backbone + ViT-based vision encoder)
  - 4-bit NF4 quantization: LLM backbone only ~1.5GB VRAM (4-bit); total with
    unquantized ViT vision encoder ~4GB VRAM. Both fit in 4070 8GB alongside
    the FHE pipeline. The ~1.5GB figure (cited in fpavlm/llm/__init__.py) refers
    to the LLM backbone only; the ~4GB figure includes the unquantized ViT.
  - Strong general VLM performance (DocVQA 90.1, ChartQA 73.5, OCRBench 845)
  - Active maintenance (Alibaba Tongyi team + HuggingFace)
  - Plug-and-play compatible with FHEViT framework (image token injection via
    masked_scatter at image_token_id positions)

PLUG-AND-PLAY CONTRACT:
  - Input: [1, N_img, llm_hidden_dim] = [1, 256, 1536] from MedicalConceptAdapter
    (Perceiver expands 16 → 256 tokens, llm_hidden_dim=1536 for Qwen2-VL-2B)
  - Output: medical report text
  - Image embedding injection: masked_scatter at image_token_id positions
    (Qwen2-VL uses <|image_pad|> token placeholder, expanded to N_img positions)
  - Chat template: Qwen2-VL's standard chat template via apply_chat_template

v167 INTEGRATION:
  - Registered as "qwen2vl-2b" in LLMRegistry
  - FHEConfig: perceiver_name="v162-default" works (PerceiverDecoder llm_dim=1536)
  - 4-bit quantization via BitsAndBytesConfig (NF4, double quant, bfloat16 compute)
  - Loads from LOCAL path only (no HuggingFace download at runtime)

USAGE:
  ```python
  from fpavlm import FHEConfig, FHEViTPipeline

  config = FHEConfig(llm_hidden_dim=1536)  # Qwen2-VL-2B hidden dim
  pipeline = FHEViTPipeline.from_pretrained(
      llm_name="qwen2vl-2b",
      llm_model_path="models/qwen2vl-2b",
      encoder_weights="artifacts/encoder_weights.pt",
      medical_priors="artifacts/medical_priors.pt",
      cfp_sensitivity="artifacts/cfp_sensitivity.pt",
      config=config,
  )
  report = pipeline.run("data/openi_subset/images/openi_00001.png")
  ```

NOTE ON EMBEDDING ADAPTATION:
  The shared encoder emits 2560-dimensional visual tokens. The required trained
  adapter at artifacts/adapters/qwen2vl-2b.pt maps them into Qwen2-VL's
  1536-dimensional native visual space; zero-shot PCA is ablation-only.

DOWNLOAD INSTRUCTIONS (user must pre-download to models/qwen2vl-2b/):
  Option 1 (HuggingFace): huggingface-cli download Qwen/Qwen2-VL-2B-Instruct \\
      --local-dir models/qwen2vl-2b --local-dir-use-symlinks False
  Option 2 (ModelScope): modelscope download --model Qwen/Qwen2-VL-2B-Instruct \\
      --local_dir models/qwen2vl-2b

REFERENCES:
  - Qwen2-VL paper: Wang et al. "Qwen2-VL: Enhancing Vision-Language Model's
    Perception of the World at Any Resolution" (arXiv 2409.12191, 2024)
  - HF docs: https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct
"""
from __future__ import annotations

from typing import List, Optional

import torch

from fpavlm.llm.interface import LLMInterface, LLMRegistry

# Default medical prompt (overridable via generate(prompt=...))
DEFAULT_MEDICAL_PROMPT_QWEN = (
    "You are a radiologist examining a chest X-ray. "
    "Write a brief radiology report. "
    "Describe the lungs (clear or abnormal), heart size, mediastinum, "
    "diaphragm, pleural spaces, and any pathology such as "
    "consolidation, effusion, or pneumothorax."
)


@LLMRegistry.register("qwen2vl-2b")
class Qwen2VLAdapter(LLMInterface):
    """Qwen2-VL-2B LLM adapter (4-bit NF4 quantization for 4070 8GB).

    Implements LLMInterface for Qwen2-VL-2B-Instruct. Loads from local path
    with 4-bit quantization. Image embeddings are injected via masked_scatter
    at <|image_pad|> token positions.

    Qwen2-VL uses a naive dynamic resolution approach: image is split into
    variable-length token sequences. The adapter replaces the variable-length
    image token sequence with the MedicalConceptAdapter output (256 tokens,
    16→256 expansion) for fixed-size FHE-compliant inference.
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
        self._llm_hidden_dim = 1536  # Qwen2-VL-2B hidden dim (Qwen2-1.5B)
        self._image_token_id = None
        self._image_token_str = "<|image_pad|>"  # Qwen2-VL image token placeholder
        self._loaded = False

    def load(self) -> None:
        """Load Qwen2-VL-2B-Instruct in 4-bit NF4 quantization.

        Uses BitsAndBytesConfig for 4-bit quantization:
          - load_in_4bit=True
          - bnb_4bit_quant_type="nf4"
          - bnb_4bit_compute_dtype=torch.bfloat16
          - bnb_4bit_use_double_quant=True

        Qwen2-VL loads via AutoModelForVision2Seq (Qwen2VLForConditionalGeneration).
        """
        from transformers import (
            AutoProcessor,
            AutoTokenizer,
            BitsAndBytesConfig,
            Qwen2VLForConditionalGeneration,
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

        # Qwen2-VL loads via Qwen2VLForConditionalGeneration
        # v174 FIX: Use device_map={"": "cuda:0"} instead of "auto" to force
        # ALL layers on GPU. device_map="auto" may place the embedding layer on
        # CPU, causing "_native_layer_norm" device mismatch during generate().
        # Qwen2-VL-2B 4-bit NF4 is ~1.5GB — fits easily on 4070 8GB.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            device_map={"": "cuda:0"},
        )
        self.model.eval()

        # Load tokenizer (Qwen2 uses Qwen2Tokenizer fast)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=False, use_fast=True
        )
        # Load processor (handles image preprocessing + tokenization)
        self.processor = AutoProcessor.from_pretrained(
            self.model_path, trust_remote_code=False
        )
        # v174 FIX: Constrain image resolution to avoid OOM and slow processing
        # on 8GB GPU. max_pixels=512×512 limits visual tokens to ~320.
        self.processor.image_processor.min_pixels = 4 * 28 * 28
        self.processor.image_processor.max_pixels = 512 * 512

        # Cache image token info
        # Qwen2-VL uses <|image_pad|> as image placeholder (single token, repeated N times)
        self._image_token_id = self.tokenizer.convert_tokens_to_ids(self._image_token_str)
        if self._image_token_id == self.tokenizer.unk_token_id:
            # Fallback: try alternative token strings
            for tok_str in ["<|image_pad|>", "<|image|>", "<image>", "<image_pad>"]:
                tid = self.tokenizer.convert_tokens_to_ids(tok_str)
                if tid != self.tokenizer.unk_token_id:
                    self._image_token_id = tid
                    self._image_token_str = tok_str
                    break

        # Verify embedding layer dtype
        embed_dtype = self.model.get_input_embeddings().weight.dtype
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
            raise RuntimeError("Qwen2VLAdapter not loaded. Call load() first.")
        return self._image_token_id

    def expected_image_token_count(self) -> int:
        """Qwen2-VL default image token count (variable, depends on resolution).

        Default 256 (matches our PerceiverDecoder 16→256 expansion).
        Qwen2-VL supports variable image token counts via <|image_pad|> repetition.
        """
        return 256

    def get_native_feature_stats(self) -> Optional[torch.Tensor]:
        """Get native vision feature statistics for adapter calibration.

        v174 FIX: Use self.processor to correctly patchify a dummy image.
        Qwen2-VL's model.visual expects pre-patchified pixel_values (a flat
        blob of num_patches × 1176 values) + image_grid_thw, NOT a raw
        [1,3,224,224] image. Passing a raw image produces 128 patches while
        image_grid_thw declares 196 (14×14), causing a tensor size mismatch
        in apply_rotary_pos_emb_vision.

        Also, model.visual's forward() ALREADY applies self.merger at the end,
        so we must NOT call self.model.merger again (it would fail with a
        reshape error on the already-merged 1536-dim features).

        Returns:
            [N_patches/4, llm_hidden_dim] native vision features, or None.
        """
        if not self._loaded:
            return None
        try:
            import numpy as np
            from PIL import Image
            with torch.no_grad():
                # Create a dummy image and use the processor to patchify it
                # correctly (produces paired pixel_values + image_grid_thw)
                dummy_img = Image.fromarray(
                    (np.random.rand(224, 224, 3) * 200 + 25).astype(np.uint8)
                )
                messages = [{"role": "user", "content": [
                    {"type": "image", "image": dummy_img},
                    {"type": "text", "text": "."},
                ]}]
                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.processor(
                    text=[text], images=[dummy_img], return_tensors="pt"
                ).to("cuda", dtype=torch.bfloat16)
                # model.visual already applies merger at the end of its forward()
                visual_features = self.model.visual(
                    inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]
                )
                return visual_features  # [N_patches/4, 1536]
        except Exception as e:
            print(f"NOTE: get_native_feature_stats for Qwen2-VL failed: {e}")
            print("      Qwen2-VL will use PCA features without calibration (works without it).")
            return None

    def extract_native_visual_features(self, image_path: str) -> Optional[torch.Tensor]:
        """Run Qwen2-VL's native vision tower on a real chest X-ray."""
        if not self._loaded:
            return None
        from PIL import Image

        try:
            with Image.open(image_path) as image:
                rgb = image.convert("RGB")
            messages = [{"role": "user", "content": [
                {"type": "image", "image": rgb},
                {"type": "text", "text": "."},
            ]}]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.processor(
                text=[text], images=[rgb], return_tensors="pt"
            ).to("cuda", dtype=torch.bfloat16)
            with torch.no_grad():
                features = self.model.visual(
                    inputs["pixel_values"], grid_thw=inputs["image_grid_thw"]
                )
            if features.dim() == 2:
                features = features.unsqueeze(0)
            return features
        except Exception as exc:
            print(f"WARNING: extract_native_visual_features failed: {exc}")
            return None

    def _build_inputs_embeds(
        self,
        image_embeds: torch.Tensor,
        prompt: str,
    ) -> tuple:
        """Build (input_ids, attention_mask, inputs_embeds) for generate.

        Qwen2-VL chat template format:
          <|im_start|>user
          <|vision_start|><|image_pad|><|vision_end|>{prompt}<|im_end|>
          <|im_start|>assistant

        The <|image_pad|> token is repeated N_img times, and the corresponding
        positions in inputs_embeds are filled with image_embeds via masked_scatter.

        Args:
            image_embeds: [1, N_img, 1536] adapter output (B=1 only).
            prompt: Text prompt for generation.

        Returns:
            (input_ids, attention_mask, inputs_embeds) on CUDA.
        """
        if not self._loaded:
            raise RuntimeError("Qwen2VLAdapter not loaded. Call load() first.")

        if image_embeds.dim() != 3 or image_embeds.shape[0] != 1:
            raise ValueError(
                f"image_embeds must be [1, N_img, {self._llm_hidden_dim}] (B=1 only), "
                f"got shape {list(image_embeds.shape)}"
            )

        # Ensure image_embeds is on CUDA + bfloat16
        if image_embeds.device.type != "cuda":
            image_embeds = image_embeds.to("cuda")
        if image_embeds.dtype != torch.bfloat16:
            image_embeds = image_embeds.to(torch.bfloat16)

        n_img_tokens = image_embeds.shape[1]

        # Build Qwen2-VL chat template
        # Qwen chat format with N image-pad tokens inside vision delimiters.
        image_seq = "".join([self._image_token_str] * n_img_tokens)
        text = (
            f"<|im_start|>user\n<|vision_start|>{image_seq}<|vision_end|>"
            f"{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )

        # Tokenize
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = enc["input_ids"].to("cuda")
        attention_mask = enc["attention_mask"].to("cuda")

        # Verify image token count
        actual_n_img = (input_ids == self._image_token_id).sum().item()
        if actual_n_img != n_img_tokens:
            # Handle mismatch (truncate or pad image_embeds)
            if actual_n_img < n_img_tokens:
                image_embeds = image_embeds[:, :actual_n_img, :]
                n_img_tokens = actual_n_img
            else:
                # More image tokens than embeds — pad with mean embedding.
                # v170 FIX: was `image_embeds[:, -1:, :]` (last token repeat)
                # which contradicted the comment and biased toward the
                # bottom-right spatial location. Mean is a neutral pad.
                pad_count = actual_n_img - n_img_tokens
                mean_embed = image_embeds.mean(dim=1, keepdim=True)  # [1, 1, D]
                pad_embeds = mean_embed.expand(1, pad_count, -1)
                image_embeds = torch.cat([image_embeds, pad_embeds], dim=1)
                n_img_tokens = actual_n_img

        # Build inputs_embeds with image embeddings injected via masked_scatter
        # Qwen2-VL: use model.get_input_embeddings()
        inputs_embeds = self.model.get_input_embeddings()(input_ids).to(torch.bfloat16)
        special_image_mask = (
            (input_ids == self._image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
        )
        embeds_flat = image_embeds.reshape(-1).contiguous().to(torch.bfloat16)
        assert embeds_flat.numel() == n_img_tokens * self._llm_hidden_dim, (
            f"embeds flat numel={embeds_flat.numel()}, "
            f"expected {n_img_tokens * self._llm_hidden_dim}"
        )
        inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, embeds_flat)

        return input_ids, attention_mask, inputs_embeds

    def embed_image_tokens(self, image_embeds: torch.Tensor, prompt: Optional[str] = None) -> torch.Tensor:
        """Convert adapter output to LLM inputs_embeds format.

        Args:
            image_embeds: [1, N_img, 1536] adapter output (B=1 only).
            prompt: Optional text prompt (defaults to DEFAULT_MEDICAL_PROMPT_QWEN).

        Returns:
            inputs_embeds: [1, seq_len, 1536] bfloat16 tensor on CUDA.
        """
        prompt = prompt if prompt is not None else DEFAULT_MEDICAL_PROMPT_QWEN
        _, _, inputs_embeds = self._build_inputs_embeds(image_embeds, prompt)
        return inputs_embeds

    def generate(
        self,
        image_embeds: torch.Tensor,
        prompt: str = DEFAULT_MEDICAL_PROMPT_QWEN,
        max_new_tokens: int = 200,
        do_sample: bool = False,
        num_beams: int = 1,
        seed: Optional[int] = 42,
        logits_processor: Optional[List] = None,
        num_return_sequences: int = 1,
    ) -> str:
        """Generate medical report from image embeddings + text prompt.

        Args:
            image_embeds: [1, N_img, 1536] adapter output.
            prompt: Text prompt.
            max_new_tokens: Maximum generation length.
            do_sample: If True, use sampling; if False, greedy/beam.
            num_beams: Number of beams (1=greedy). Beam search is more robust
                to ~1e-5 FHE noise.
            seed: Random seed for reproducibility.
            logits_processor: v168 — Optional list of HuggingFace-compatible
                LogitsProcessor objects (e.g., TopNSigmaLogitProcessor from
                fpavlm.llm.noise_aware_decoding). When provided, these are
                passed to model.generate() to apply per-step logit denoising
                (v166 Innovation 23 — actively deployed in v168).
            num_return_sequences: v168 — Number of sequences to generate per
                prompt. Used by MBR decoding. Default 1.

        Returns:
            Generated text string (medical report).
        """
        if not self._loaded:
            raise RuntimeError("Qwen2VLAdapter not loaded. Call load() first.")

        # v174 FIX: Handle prompt=None from pipeline.run() (which defaults to None
        # meaning "use LLM default"). Without this, None overrides the default
        # parameter and _build_inputs_embeds receives None, producing generic output
        # instead of a radiology report.
        prompt = prompt if prompt is not None else DEFAULT_MEDICAL_PROMPT_QWEN

        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)

        input_ids, attention_mask, inputs_embeds = self._build_inputs_embeds(image_embeds, prompt)

        # v168: Build generation kwargs, conditionally adding logits_processor.
        gen_kwargs = dict(
            input_ids=None,  # Use inputs_embeds instead
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            num_beams=num_beams,
            repetition_penalty=1.15,
            no_repeat_ngram_size=6,
            temperature=0.7 if do_sample else 1.0,
            top_p=0.9 if do_sample else 1.0,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )
        if logits_processor is not None and len(logits_processor) > 0:
            gen_kwargs["logits_processor"] = logits_processor
        if num_return_sequences > 1:
            gen_kwargs["num_return_sequences"] = num_return_sequences
            gen_kwargs["num_beams"] = max(num_beams, num_return_sequences)

        with torch.no_grad():
            # Qwen2-VL generate uses model.generate (delegates to language_model)
            output = self.model.generate(**gen_kwargs)

        # v168fix: When generate() is called with `input_ids=None, inputs_embeds=...`,
        # HuggingFace returns ONLY the newly-generated tokens — the output tensor
        # does NOT include the input prefix. The previous code sliced
        # `output[0, inputs_embeds.shape[1]:]` which indexed PAST the end of the
        # output tensor, yielding an empty tensor and an empty decoded string.
        #
        # Root cause: the output tensor contains generated tokens only.
        #
        # FIX: Use the full output tensor directly. output shape is
        # [1 (or nrs), max_new_tokens] — no input prefix to skip. When
        # num_return_sequences > 1, only the first sequence is returned.
        generated_ids = output[0, :]
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return text


__all__ = ["Qwen2VLAdapter"]
