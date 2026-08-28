# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

from typing import ClassVar, cast
import json
import os

import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.platforms import current_platform
from vllm.transformers_utils.configs.qwen3_8_flash_next import (
    Qwen3_8FlashNextTextConfig,
)
from vllm.utils.torch_utils import (
    LayerNameType,
    _encode_layer_name,
    _resolve_layer_name,
    canonicalize_singleton_dim_strides,
    direct_register_custom_op,
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer
from vllm.logger import init_logger

logger = init_logger(__name__)


def _apply_qsa_fp8_scales(layer: nn.Module) -> None:
    """Load static per-tensor K/V scales for the FP8 QSA main cache.

    Sources, in priority order:
    1. QSA_FP8_SCALES_FILE: JSON ``{"k_scale": f, "v_scale": f}`` applied to
       every layer, or ``{"per_layer": {"<layer_name>": {"k_scale": f,
       "v_scale": f}, ...}, "default": {"k_scale": f, "v_scale": f}}``.
    2. QSA_FP8_K_SCALE + QSA_FP8_V_SCALE environment floats (both required).
    3. Fallback 1.0 with an explicit non-quality warning.

    A requested but unreadable/incomplete calibrated source is a hard error;
    silent fallback to 1.0 would corrupt quality measurements (VALIDATION.md).
    """
    scales_file = os.environ.get("QSA_FP8_SCALES_FILE")
    k_scale = v_scale = None
    if scales_file:
        with open(scales_file) as f:  # hard error if missing: intentional
            data = json.load(f)
        entry = None
        per_layer = data.get("per_layer")
        if per_layer and layer.layer_name in per_layer:
            entry = per_layer[layer.layer_name]
        elif "default" in data:
            entry = data["default"]
        elif "k_scale" in data:
            entry = data
        if entry is None or "k_scale" not in entry or "v_scale" not in entry:
            raise ValueError(
                f"QSA_FP8_SCALES_FILE={scales_file} has no k_scale/v_scale "
                f"entry for layer {layer.layer_name}"
            )
        k_scale, v_scale = float(entry["k_scale"]), float(entry["v_scale"])
        source = f"file {scales_file}"
    else:
        env_k = os.environ.get("QSA_FP8_K_SCALE")
        env_v = os.environ.get("QSA_FP8_V_SCALE")
        if (env_k is None) != (env_v is None):
            raise ValueError(
                "QSA_FP8_K_SCALE and QSA_FP8_V_SCALE must be set together"
            )
        if env_k is not None:
            k_scale, v_scale = float(env_k), float(env_v)
            source = "environment"
    if k_scale is None:
        k_scale = v_scale = 1.0
        logger.warning(
            "QSA FP8 KV cache for %s uses scale=1.0 (non-quality bring-up "
            "mode; provide QSA_FP8_SCALES_FILE for calibrated runs)",
            layer.layer_name,
        )
    else:
        logger.info(
            "QSA FP8 KV scales for %s from %s: k=%.6g v=%.6g",
            layer.layer_name, source, k_scale, v_scale,
        )
    if k_scale <= 0 or v_scale <= 0:
        raise ValueError("QSA FP8 scales must be positive")
    layer._k_scale.fill_(k_scale)
    layer._v_scale.fill_(v_scale)
    layer._k_scale_float = float(k_scale)
    layer._v_scale_float = float(v_scale)

# --- Optional calibration collection (QSA_FP8_CALIBRATE_OUT) --------------
# Opt-in running absmax collector for K/V rows entering the cache. Active
# only when QSA_FP8_CALIBRATE_OUT is set; intended for dedicated calibration
# runs, never for production. GPU-resident running maxima (no per-step sync);
# flushed to disk periodically and at exit.
_CALIBRATE_OUT = os.environ.get("QSA_FP8_CALIBRATE_OUT")
_CALIB_STATE: dict[str, dict[str, torch.Tensor | int]] = {}
_CALIB_FLUSH_EVERY = 50  # forward calls


def _calibration_update(
    layer: nn.Module, key: torch.Tensor, value: torch.Tensor
) -> None:
    if _CALIBRATE_OUT is None:
        return
    entry = _CALIB_STATE.get(layer.layer_name)
    if entry is None:
        entry = _CALIB_STATE[layer.layer_name] = {
            "k_absmax": torch.zeros((), dtype=torch.float32, device=key.device),
            "v_absmax": torch.zeros((), dtype=torch.float32, device=key.device),
            "calls": 0,
        }
    # Warmup/capture passes can carry NaN/inf dummy data; only finite
    # magnitudes may influence a calibration maximum.
    k_now = torch.nan_to_num(key.detach().float().abs().max(), nan=0.0,
                             posinf=0.0)
    v_now = torch.nan_to_num(value.detach().float().abs().max(), nan=0.0,
                             posinf=0.0)
    torch.maximum(entry["k_absmax"], k_now, out=entry["k_absmax"])
    torch.maximum(entry["v_absmax"], v_now, out=entry["v_absmax"])
    entry["calls"] += 1
    if entry["calls"] % _CALIB_FLUSH_EVERY == 0:
        _calibration_flush()


def _calibration_flush() -> None:
    if _CALIBRATE_OUT is None or not _CALIB_STATE:
        return
    from vllm.distributed import get_tensor_model_parallel_rank

    rank = get_tensor_model_parallel_rank()
    per_layer = {}
    for name, entry in sorted(_CALIB_STATE.items()):
        k_abs = float(entry["k_absmax"].item())
        v_abs = float(entry["v_absmax"].item())
        per_layer[name] = {
            "k_absmax": k_abs,
            "v_absmax": v_abs,
            "k_scale": k_abs / 448.0,
            "v_scale": v_abs / 448.0,
        }
    path = f"{_CALIBRATE_OUT}.rank{rank}.json"
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"per_layer": per_layer}, f, indent=2)
    os.replace(tmp, path)


if _CALIBRATE_OUT is not None:
    import atexit

    atexit.register(_calibration_flush)
    logger.info("QSA FP8 calibration collection active -> %s", _CALIBRATE_OUT)


class Qwen3_8FlashNextQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen3_8FlashNextQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
    ]

    # FP8 E4M3 is supported as *storage* with a software decode in the QSA
    # Triton kernel. The inherited FlashAttention classmethods route every
    # quantized dtype through flash_attn_supports_kv_cache_dtype, which is
    # sm_90/100-only; both gates are overridden here to check list membership
    # only. Generic FlashAttention semantics are untouched.
    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype is None:
            return True
        return kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_combination(
        cls,
        head_size,
        dtype,
        kv_cache_dtype,
        block_size,
        use_mla,
        has_sink,
        use_sparse,
        use_mm_prefix,
        device_capability,
    ):
        # Skip only the FP8 arch gate; all other FA combination rules apply.
        return FlashAttentionBackend.supports_combination.__func__(
            cls,
            head_size,
            dtype,
            None if kv_cache_dtype in ("fp8", "fp8_e4m3") else kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            device_capability,
        )

    @staticmethod
    def get_name() -> str:
        return "QWEN38_FLASH_NEXT_QSA_TRITON"

    @staticmethod
    def get_impl_cls() -> type[Qwen3_8FlashNextQSAFlashAttentionImpl]:
        return Qwen3_8FlashNextQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen3_8FlashNextQSAMetadataBuilder]:
        return Qwen3_8FlashNextQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen3_8FlashNextQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(self, *args, **kwargs) -> None:
        # FlashAttentionImpl.__init__ rejects FP8 KV caches unless the
        # platform is sm_90/sm_100. QSA never runs the FA kernels — it only
        # inherits do_kv_cache_update and metadata plumbing — so the dtype is
        # masked during super().__init__ and restored immediately after.
        requested_kv_dtype = args[6] if len(args) > 6 else kwargs.get(
            "kv_cache_dtype", "auto"
        )
        if requested_kv_dtype in ("fp8", "fp8_e4m3"):
            if len(args) > 6:
                args = (*args[:6], "auto", *args[7:])
            else:
                kwargs["kv_cache_dtype"] = "auto"
        super().__init__(*args, **kwargs)
        self.kv_cache_dtype = requested_kv_dtype
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen3.8-Flash-Next QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 or FP8 E4M3 main KV cache"
            )
        self.supports_quant_query_input = False

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        key_cache, value_cache = kv_cache.transpose(1, 2).split(self.head_size, dim=-1)
        key_cache = canonicalize_singleton_dim_strides(key_cache)
        value_cache = canonicalize_singleton_dim_strides(value_cache)
        is_fp8 = self.kv_cache_dtype in ("fp8", "fp8_e4m3")
        expected_cache_dtype = torch.uint8 if is_fp8 else torch.bfloat16
        if key_cache.dtype != expected_cache_dtype or query.dtype != torch.bfloat16:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires BF16 Q and a "
                f"{'uint8 FP8' if is_fp8 else 'BF16'} main KV cache, got "
                f"cache={key_cache.dtype} query={query.dtype}"
            )

        from .ops.qsa import qsa_sparse_paged_attention

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            output[:num_tokens],
            k_scale=layer._k_scale if is_fp8 else None,
            v_scale=layer._v_scale if is_fp8 else None,
        )
        return output


class Qwen3_8FlashNextQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen3_8FlashNextTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen3.8-Flash-Next QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen3.8-Flash-Next QSA currently requires BF16")
        if cache_config.cache_dtype not in ("auto", "bfloat16", "fp8", "fp8_e4m3"):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires a BF16 or FP8 E4M3 main KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support KV quantization"
            )
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA requires causal decoder attention"
            )

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError(
                "Qwen3.8-Flash-Next QSA does not support dual-chunk RoPE"
            )
        # Qwen3.8-Flash-Next full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and current_platform.is_cuda()
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = cache_config.cache_dtype
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        _is_fp8_kv = self.kv_cache_dtype in ("fp8", "fp8_e4m3")
        expected_storage = torch.uint8 if _is_fp8_kv else torch.bfloat16
        if self.kv_cache_torch_dtype != expected_storage:
            raise NotImplementedError(
                f"Qwen3.8-Flash-Next QSA requires {expected_storage} cache "
                f"storage for kv_cache_dtype={self.kv_cache_dtype}"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)
        if _is_fp8_kv:
            _apply_qsa_fp8_scales(self)

        self.attn_backend = Qwen3_8FlashNextQSAFlashAttentionBackend
        self.impl = Qwen3_8FlashNextQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return FullAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_dim,
            head_size_v=self.head_dim,
            dtype=self.kv_cache_torch_dtype,
            kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
        )

    def _run_qsa(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        _calibration_update(self, key, value)
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected = self.indexer(
            hidden_states,
            positions,
            self.topk_indices_buffer[:num_tokens],
        )
        if selected.shape != (
            num_tokens,
            self.indexer.output_width,
        ):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen3_8FlashNextQSAFlashAttentionImpl, self.impl)
        impl.do_kv_cache_update(
            self,
            key,
            value,
            self.kv_cache,
            main_metadata.slot_mapping,
        )
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v, gate = self._project_qkv_gate(qkv, positions)
        num_tokens = hidden_states.shape[0]
        query = q.view(num_tokens, self.num_heads, self.head_dim)
        key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        attn_output = torch.empty_like(query)
        encoded_layer_name = _encode_layer_name(self.layer_name)
        if current_platform.opaque_attention_op():
            torch.ops.vllm.qwen3_8_flash_next_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        else:
            qwen3_8_flash_next_qsa_with_output(
                hidden_states,
                positions,
                query,
                key,
                value,
                attn_output,
                encoded_layer_name,
            )
        flat_output = attn_output.view(num_tokens, -1)
        if gate is not None:
            flat_output = flat_output * torch.sigmoid(gate)
        output, _ = self.o_proj(flat_output)
        return output


def qwen3_8_flash_next_qsa_with_output(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    """Run the complete QSA state/update/attend transaction."""

    layer_name = _resolve_layer_name(layer_name)
    layer = get_forward_context().no_compile_layers[layer_name]
    if not isinstance(layer, Qwen3_8FlashNextQSAAttention):
        raise TypeError(f"{layer_name} is not a Qwen3.8-Flash-Next QSA owner")
    layer._run_qsa(
        hidden_states,
        positions,
        query,
        key,
        value,
        output,
    )


def qwen3_8_flash_next_qsa_with_output_fake(
    hidden_states: torch.Tensor,
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: LayerNameType,
) -> None:
    del hidden_states, positions, query, key, value, output, layer_name


direct_register_custom_op(
    op_name="qwen3_8_flash_next_qsa_with_output",
    op_func=qwen3_8_flash_next_qsa_with_output,
    mutates_args=["output"],
    fake_impl=qwen3_8_flash_next_qsa_with_output_fake,
)


__all__ = [
    "QSAIndexer",
    "Qwen3_8FlashNextQSAAttention",
    "Qwen3_8FlashNextQSAFlashAttentionBackend",
    "Qwen3_8FlashNextQSAFlashAttentionImpl",
    "qwen3_8_flash_next_qsa_with_output",
]
