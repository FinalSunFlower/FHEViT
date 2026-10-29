"""Client-assisted CKKS refresh and nonlinear services.

CANR decrypts and re-encrypts an exhausted ciphertext on the data owner's
device. CAS additionally returns an encryption of the elementwise square, and
CADN returns an encryption of exact LayerNorm output. These services reset the
CKKS level and avoid selected polynomial approximations, but require an
interactive client round trip.

The server context never contains the secret key, so the server cannot observe
the image or intermediate plaintext. The client necessarily observes every
refreshed intermediate and public normalization affine parameters. Repeated
chosen-input observations can leak information about server weights; optional
Gaussian masking is an empirical mitigation, not a cryptographic model-privacy
guarantee. Deployments requiring model confidentiality must use the separately
trained callback-free polynomial profile.

The default research parameters use ``N=32768``, a 28-bit scale, and 13
multiplicative levels. With TenSEAL automatic rescaling, ciphertext-ciphertext,
ciphertext-plaintext, and ciphertext-scalar multiplication each consume one
level. Security and precision claims must therefore be taken from the recorded
parameter artifact and measured CKKS run, not inferred from this module.
"""
from __future__ import annotations

import time
from typing import Optional

import torch


class ClientCANREngine:
    """Client-side CANR Engine: holds secret_key, can encrypt + decrypt.

    This is the ONLY entity that can decrypt. Server never gets secret_key.
    Client compute is limited: only encrypt/decrypt, no model inference.

    The public server context is derived without the secret key. Protocol code
    must never transmit ``self.context`` or its secret-key serialization.
    """

    def __init__(self, poly_modulus_degree: int = 32768,
                 scale_bits: int = 28, depth_budget: int = 13,
                 *, generate_relin_keys: bool = True):
        """Initialize Client CANR engine with TenSEAL CKKS context.

        Args:
            poly_modulus_degree: CKKS polynomial modulus degree (N)
            scale_bits: CKKS scale in bits (``2**scale_bits``).
            depth_budget: Multiplicative depth budget. Default=13 gives
              coeff_mod = [60, 28*13, 60] = 484 bits < 881-bit 128-sec limit
              for ``N=32768``. The runtime smoke test records actual parameters.
            generate_relin_keys: Generate relinearization keys for
              ciphertext-ciphertext products. Degree-1 callback-free profiles
              can set this to ``False``; the default stays enabled for legacy
              degree-2/CAS paths.
        """
        import tenseal as ts
        self.ts = ts
        self.scale = 2 ** scale_bits
        self.slot_count = poly_modulus_degree // 2
        self.poly_modulus_degree = poly_modulus_degree
        self.scale_bits = scale_bits
        self.depth_budget = depth_budget
        self.generate_relin_keys = bool(generate_relin_keys)

        # Client creates context WITH secret_key (client owns it)
        coeff_mod_sizes = [60] + [scale_bits] * depth_budget + [60]
        self.context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=poly_modulus_degree,
            coeff_mod_bit_sizes=coeff_mod_sizes,
        )
        self.context.global_scale = self.scale
        self.context.auto_relin = self.generate_relin_keys
        self.context.auto_rescale = True
        self.context.auto_mod_switch = True
        self.context.generate_galois_keys()
        if self.generate_relin_keys:
            self.context.generate_relin_keys()

        # Keep a distinct public context for ciphertexts crossing the trust
        # boundary.  TenSEAL ciphertext objects retain the context they were
        # created with; returning a vector created from ``self.context`` would
        # therefore leak a usable secret-key handle in an in-process protocol.
        self.public_context = self.context.copy()
        self.public_context.make_context_public()

        # CRITICAL: Client holds secret_key, NEVER shares it with server
        # self.context.secret_key() exists but is never transmitted

        self.refresh_count = 0
        self.total_refresh_time = 0.0

    def encrypt(self, tensor: torch.Tensor) -> list:
        """Client encrypts plaintext tensor → list of CKKS ciphertexts.

        Args:
            tensor: [B, D] or [D] plaintext tensor

        Returns:
            List of TenSEAL CKKS ciphertexts (one per row)
        """
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2:
            raise ValueError(f"encrypt expects a rank-1 or rank-2 tensor, got {tensor.shape}")
        if tensor.shape[1] > self.slot_count:
            raise ValueError(
                f"input row has {tensor.shape[1]} values but CKKS provides only "
                f"{self.slot_count} slots"
            )
        tensor_np = tensor.detach().cpu().numpy()
        ct_list = []
        for i in range(tensor_np.shape[0]):
            ct = self.ts.ckks_vector(self.context, tensor_np[i].tolist())
            ct_list.append(self._to_public_cipher(ct))
        return ct_list

    def encrypt_chunked(self, tensor: torch.Tensor, *, chunk_size: Optional[int] = None):
        """Encrypt rows split into CKKS-slot-sized chunks.

        This is the transport primitive for raw image patches whose flattened
        dimension exceeds ``N/2``.  It returns a flat ciphertext list plus a
        layout descriptor consumed by the server's linear patch projection;
        no plaintext is exposed after this client-side operation.
        """
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 2:
            raise ValueError(f"encrypt_chunked expects rank-1/2 tensor, got {tensor.shape}")
        width = int(chunk_size or self.slot_count)
        if width < 1 or width > self.slot_count:
            raise ValueError(f"chunk_size must be in [1, {self.slot_count}], got {width}")
        rows, features = (int(tensor.shape[0]), int(tensor.shape[1]))
        parts = (features + width - 1) // width
        tensor_np = tensor.detach().cpu().numpy()
        ct_list = []
        for row in range(rows):
            for part in range(parts):
                start = part * width
                stop = min(start + width, features)
                ct = self.ts.ckks_vector(self.context, tensor_np[row, start:stop].tolist())
                ct_list.append(self._to_public_cipher(ct))
        return ct_list, {
            "token_count": rows,
            "parts_per_token": parts,
            "chunk_size": width,
            "feature_dim": features,
        }

    def _to_public_cipher(self, ct):
        """Rebind a ciphertext to the public context before transmission."""
        return self.ts.ckks_vector_from(self.public_context, ct.serialize())

    def _to_private_cipher(self, ct):
        """Rebind a received ciphertext to the client's private context."""
        return self.ts.ckks_vector_from(self.context, ct.serialize())

    def decrypt(self, ct_list: list, dim: Optional[int] = None) -> torch.Tensor:
        """Client decrypts ciphertexts → plaintext tensor.

        Only client can call this (only client has secret_key).
        Server NEVER calls this method.

        Args:
            ct_list: List of TenSEAL CKKS ciphertexts
            dim: Expected output dimension (truncate to this)

        Returns:
            Decrypted plaintext tensor [B, D]
        """
        plains = []
        d = dim if dim else self.slot_count
        for ct in ct_list:
            plain = self._to_private_cipher(ct).decrypt()
            plains.append(torch.tensor(plain[:d]))
        return torch.stack(plains, dim=0)

    def canr_refresh(self, ct_list: list) -> list:
        """CANR: Client decrypts exhausted ciphertexts and re-encrypts.

        This is the CORE CANR operation. It MUST be called on CLIENT side.
        Server sends exhausted ciphertexts here, client refreshes them.

        Protocol:
        1. Server sends exhausted ciphertexts to client (server cannot decrypt)
        2. Client decrypts (client has secret_key)
        3. Client re-encrypts with FRESH noise budget
        4. Client sends refreshed ciphertexts back to server

        Args:
            ct_list: List of exhausted TenSEAL ciphertexts (from server)

        Returns:
            List of fresh TenSEAL ciphertexts (full depth budget)
        """
        t_start = time.time()

        ct_refreshed = []
        for ct in ct_list:
            ct = self._to_private_cipher(ct)
            # Step 2: Client decrypts (only client can do this)
            plain_vals = ct.decrypt()
            # Step 3: Client re-encrypts with fresh noise budget
            ct_fresh = self.ts.ckks_vector(self.context, plain_vals)
            ct_refreshed.append(self._to_public_cipher(ct_fresh))

        t_elapsed = time.time() - t_start
        self.refresh_count += 1
        self.total_refresh_time += t_elapsed

        return ct_refreshed

    def cas_refresh(self, ct_list: list) -> tuple:
        """Refresh ``x`` and return a fresh encryption of ``x**2``.

        Computing the square after client decryption replaces a subsequent
        ciphertext-ciphertext square with ciphertext-plaintext products. The
        trade-off is one additional encryption per ciphertext and disclosure of
        the refreshed intermediate to the client.

        Args:
            ct_list: List of exhausted TenSEAL ciphertexts (from server)

        Returns:
            Tuple (ct_refreshed, ct_squared):
              - ct_refreshed: fresh ciphertexts of x (normal CANR)
              - ct_squared: fresh ciphertexts of x² (for PolyNorm's c2·x²)
        """
        import numpy as np

        t_start = time.time()

        ct_refreshed = []
        ct_squared = []
        for ct in ct_list:
            ct = self._to_private_cipher(ct)
            # Step 1: Client decrypts (only client can do this)
            plain_vals = ct.decrypt()
            plain_np = np.array(plain_vals, dtype=np.float64)
            # Step 2: Compute square in plaintext (trivial, exact)
            sq_vals = plain_np ** 2
            # Step 3: Re-encrypt original (normal CANR)
            ct_fresh = self.ts.ckks_vector(self.context, plain_vals)
            ct_refreshed.append(self._to_public_cipher(ct_fresh))
            # Step 4: Re-encrypt square (CAS — for PolyNorm's c2·x²)
            ct_sq_fresh = self.ts.ckks_vector(self.context, sq_vals.tolist())
            ct_squared.append(self._to_public_cipher(ct_sq_fresh))

        t_elapsed = time.time() - t_start
        self.refresh_count += 1
        self.total_refresh_time += t_elapsed

        return ct_refreshed, ct_squared

    def cadn_refresh(
        self,
        ct_list: list,
        norm_weight: Optional[torch.Tensor] = None,
        norm_bias: Optional[torch.Tensor] = None,
        eps: float = 1e-6,
        use_mean_subtraction: bool = True,
    ) -> tuple:
        """Refresh ``x`` and return an encryption of client-computed LayerNorm.

        LayerNorm uses mean, variance, square root, and division. Computing it
        at the refresh boundary avoids a polynomial approximation and returns a
        depth-zero ciphertext. The client learns the intermediate plus any
        supplied affine parameters, so this mode does not satisfy a malicious
        client's model-confidentiality threat model.

        Args:
            ct_list: List of exhausted TenSEAL ciphertexts (from server, N tokens × D slots)
            norm_weight: [D] LayerNorm γ parameter (PUBLIC, sent from server as plain tensor).
                If None, γ=1 (pure normalization, no affine — affine fused into next matmul).
            norm_bias: [D] LayerNorm β parameter (PUBLIC). If None, β=0.
            eps: Numerical stability for var (default 1e-6).
            use_mean_subtraction: True=LayerNorm (subtract mean), False=RMSNorm (no mean sub).
                Default True (matches SigLIP pre-training).

        Returns:
            Tuple (ct_refreshed, ct_normalized):
              - ct_refreshed: fresh ciphertexts of x (normal CANR, for next block)
              - ct_normalized: fresh ciphertexts of z_norm (depth 0)
        """
        import numpy as np

        t_start = time.time()

        # Convert norm params to numpy once (PUBLIC constants from server)
        gamma_np = norm_weight.detach().cpu().numpy() if norm_weight is not None else None
        beta_np = norm_bias.detach().cpu().numpy() if norm_bias is not None else None

        ct_refreshed = []
        ct_normalized = []
        for ct in ct_list:
            ct = self._to_private_cipher(ct)
            # Step 1: Client decrypts (only client can do this — same as CANR/CAS)
            plain_vals = ct.decrypt()
            plain_np = np.array(plain_vals, dtype=np.float64)

            # Step 2: Client computes EXACT LayerNorm in plaintext (trivial, O(D))
            #   μ = mean(x),  var = mean((x-μ)²),  z = (x-μ)/sqrt(var+ε) · γ + β
            # This is the EXACT LayerNorm — no polynomial approximation, no static baking.
            # Computed per-token (each ciphertext is one token's D-dim feature vector).
            if use_mean_subtraction:
                # LayerNorm: subtract mean, divide by std
                mu = plain_np.mean()
                centered = plain_np - mu
                var = (centered * centered).mean()
                scale = 1.0 / np.sqrt(var + eps)
                z_norm = centered * scale
            else:
                # RMSNorm: no mean subtraction (var = mean(x²))
                mean_x2 = (plain_np * plain_np).mean()
                scale = 1.0 / np.sqrt(mean_x2 + eps)
                z_norm = plain_np * scale

            # Apply affine (γ, β) — these are PUBLIC constants from server
            # (same privacy contract as CAS's c0/c1/c2 polynomial coefficients)
            if gamma_np is not None:
                z_norm = z_norm * gamma_np
            if beta_np is not None:
                z_norm = z_norm + beta_np

            # Step 3: Re-encrypt original x (normal CANR — for next block's matmul)
            ct_fresh = self.ts.ckks_vector(self.context, plain_vals)
            ct_refreshed.append(self._to_public_cipher(ct_fresh))
            # Step 4: Re-encrypt normalized z (CADN — for this block's normalization)
            ct_z_fresh = self.ts.ckks_vector(self.context, z_norm.tolist())
            ct_normalized.append(self._to_public_cipher(ct_z_fresh))

        t_elapsed = time.time() - t_start
        self.refresh_count += 1
        self.total_refresh_time += t_elapsed

        return ct_refreshed, ct_normalized

    def estimate_embedding_noise(self, embedding: torch.Tensor) -> float:
        """Estimate fresh re-encryption noise on a decrypted embedding.

        This diagnostic is a lower bound, not accumulated circuit error. Deployment
        decoding must use paired FHE/plain release calibration instead.

        METHOD (re-encryption differential):
          1. Take the decrypted embedding x (16×2560 plain tensor)
          2. Re-encrypt x → x_ct_fresh (fresh ciphertext, depth 0)
          3. Decrypt x_ct_fresh → x_redec
          4. σ_fhe = std(x - x_redec)

        This measures the CKKS encryption/decryption noise floor. It captures:
          - The rounding noise from CKKS encoding (scale=2^28 → ~2^-28 ≈ 3.7e-9)
          - The NTT/INTT rounding noise
          - The slot-to-coefficient mapping noise

        It does not include errors accumulated before the release boundary and
        must not be presented as the release error distribution.

        Args:
            embedding: [16, D] or [B, 16, D] plain decrypted tensor.

        Returns:
            σ_fhe (float): measured CKKS noise std on the embedding.
        """
        import numpy as np

        # Per-token encryption prevents a flattened release from exceeding the
        # CKKS slot count.
        if embedding.dim() == 3:
            # [B, 16, D] → take first batch
            embedding = embedding[0]
        # embedding is now [16, D]

        noises = []
        for i in range(embedding.shape[0]):
            x_row = embedding[i].detach().cpu().numpy().astype(np.float64)
            # Re-encrypt this single token → decrypt → measure differential
            ct_fresh = self.ts.ckks_vector(self.context, x_row.tolist())
            x_redec = np.array(ct_fresh.decrypt()[:len(x_row)], dtype=np.float64)
            noise = x_row - x_redec
            noises.append(noise)

        # Aggregate noise across all tokens
        all_noise = np.concatenate(noises)
        sigma_fhe = float(np.std(all_noise))

        # Track for stats
        self.last_noise_estimate = sigma_fhe

        return sigma_fhe

    def get_stats(self) -> dict:
        return {
            'refresh_count': self.refresh_count,
            'total_time': self.total_refresh_time,
            'avg_time_per_refresh': self.total_refresh_time / max(self.refresh_count, 1),
            'last_noise_estimate': getattr(self, 'last_noise_estimate', None),
        }


class ServerCANREngine:
    """Server-side CANR Engine: NO secret_key, can only do homomorphic ops.

    Privacy guarantee:
    - Server can perform cipher×plain, cipher×cipher, cipher+cipher
    - Server CANNOT decrypt (no secret_key in context)
    - Server sends exhausted ciphertexts to client for CANR refresh
    - Server NEVER sees plaintext

    This enforces the two-setting requirement:
    - Server holds model params, does most computation (FHE-based)
    - Server cannot access raw image data (no decrypt capability)
    """

    def __init__(self, client_engine: ClientCANREngine):
        """Initialize Server CANR engine from client's context (PUBLIC part only).

        Args:
            client_engine: ClientCANREngine (server gets PUBLIC context, NOT secret_key)
        """
        import tenseal as ts
        self.ts = ts

        # CRITICAL: Server gets a COPY of context WITHOUT secret_key
        # make_context_public() strips secret_key so server cannot decrypt
        self.context = client_engine.context.copy()
        self.context.make_context_public()  # Remove secret_key from server's context

        self.scale = client_engine.scale
        self.slot_count = client_engine.slot_count

    def matmul_plain(self, ct_list: list, weight) -> list:
        """Server: cipher × plain matmul (consumes 1 depth in TenSEAL)."""
        weight_np = weight.detach().cpu().numpy() if hasattr(weight, 'detach') else weight
        return [ct.matmul(weight_np) for ct in ct_list]

    def add_gaussian_noise(self, ct_list: list, sigma: float, data_dim: int) -> list:
        """Add empirical Gaussian masking before an interactive CADN boundary.

        MODEL STEALING DEFENSE (Gaussian masking of the weight-bearing signal):
          Without this, the client receives Y = X·W in plaintext during CADN and
          can recover W with D queries (verified: error 5e-14). This method adds
          fresh N(0, sigma^2) masking noise in the ENCRYPTED domain (cipher + plain,
          0 depth) before the ciphertext leaves the server. The client decrypts
          Y+N, computes LayerNorm(Y+N) — the normalization suppresses small noise
          so cos impact is negligible, but W recovery error becomes ~sigma*sqrt(D/N).

        Terminology note: the masking noise yields a DP-style recovery bound
        (sigma*sqrt(D/N)) on the server's weights, but the protected asset is the
        model IP (weights), not training-data individuals — hence "Gaussian masking"
        rather than "DP noise". The config flag `pre_cadn_noise_sigma` is retained
        for API stability.

        This is the server-side primitive. The encoder's _apply_pre_cadn_noise()
        is the production path (uses self.dim directly); this method is for unit
        tests and standalone server-side usage.

        Args:
            ct_list: List of TenSEAL CKKS ciphertexts.
            sigma: Masking-noise standard deviation; select it from a recorded
                utility/model-extraction calibration rather than this API.
            data_dim: Number of VALUES per ciphertext (NOT slot_count). TenSEAL
                requires the plain vector to match the ciphertext's logical size.
                For FHE-ViT CADN, this is the encoder dim (e.g., 64).

        Returns:
            List of masked ciphertexts (0 depth added — addition is free in CKKS).
        """
        import numpy as np
        noised = []
        for ct in ct_list:
            noise = (np.random.randn(data_dim) * sigma).tolist()
            noised.append(ct + noise)  # cipher + plain: 0 depth
        return noised
