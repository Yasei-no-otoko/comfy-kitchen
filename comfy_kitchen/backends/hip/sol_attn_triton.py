"""ROCm Sol-Attn using native HIP quantization and a Triton sparse forward."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from comfy_kitchen.constraints import sol_attn_common_call_rule

BLOCK = 64
GROUP_PAD = 64
LOG2E = 1.4426950408889634
BLOCK_TL = tl.constexpr(BLOCK)
LOG2E_TL = tl.constexpr(LOG2E)


@triton.jit
def _reduce_kc_kernel(
    k_ptr,
    kc_ptr,
    tokens,
    stride_b,
    stride_t,
    stride_h,
    heads: tl.constexpr,
    blocks_padded: tl.constexpr,
    head_dim: tl.constexpr,
):
    block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    rows = block * BLOCK_TL + tl.arange(0, BLOCK_TL)
    dims = tl.arange(0, head_dim)
    values = tl.load(
        k_ptr
        + batch * stride_b
        + rows[:, None].to(tl.int64) * stride_t
        + head * stride_h
        + dims[None, :],
        mask=(rows < tokens)[:, None],
        other=0.0,
    ).to(tl.float32)
    block_len = tl.minimum(BLOCK_TL, tokens - block * BLOCK_TL).to(tl.float32)
    summary = tl.sum(values, axis=0) / block_len
    tl.store(
        kc_ptr + ((batch * blocks_padded + block) * heads + head) * head_dim + dims,
        summary,
    )


@triton.jit
def _reduce_vc_kernel(
    v_ptr,
    vc_ptr,
    tokens,
    stride_b,
    stride_t,
    stride_h,
    heads: tl.constexpr,
    blocks_padded: tl.constexpr,
    head_dim: tl.constexpr,
):
    block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    rows = block * BLOCK_TL + tl.arange(0, BLOCK_TL)
    dims = tl.arange(0, head_dim)
    values = tl.load(
        v_ptr
        + batch * stride_b
        + rows[:, None].to(tl.int64) * stride_t
        + head * stride_h
        + dims[None, :],
        mask=(rows < tokens)[:, None],
        other=0.0,
    ).to(tl.float32)
    summary = tl.sum(values, axis=0)
    tl.store(
        vc_ptr + ((batch * blocks_padded + block) * heads + head) * head_dim + dims,
        summary,
    )


@triton.jit
def _center_kc_kernel(
    k_ptr,
    anchor_indices_ptr,
    kc_ptr,
    stride_b,
    stride_t,
    stride_h,
    heads: tl.constexpr,
    blocks_padded: tl.constexpr,
    head_dim: tl.constexpr,
):
    block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    dims = tl.arange(0, head_dim)
    anchor_index = tl.load(anchor_indices_ptr + batch_head)
    anchor = tl.load(
        k_ptr
        + batch * stride_b
        + tl.maximum(anchor_index, 0).to(tl.int64) * stride_t
        + head * stride_h
        + dims,
    ).to(tl.float32)
    anchor = tl.where(anchor_index >= 0, anchor, 0.0)
    offset = ((batch * blocks_padded + block) * heads + head) * head_dim + dims
    centroid = tl.load(kc_ptr + offset).to(tl.float32)
    tl.store(kc_ptr + offset, centroid - anchor)


@triton.jit
def _threshold_kernel(
    q_ptr,
    kc_mean_ptr,
    kc_var_ptr,
    threshold_ptr,
    tau,
    scale,
    tokens,
    stride_b,
    stride_t,
    stride_h,
    heads: tl.constexpr,
    blocks: tl.constexpr,
    head_dim: tl.constexpr,
):
    q_block = tl.program_id(0)
    batch_head = tl.program_id(1)
    batch = batch_head // heads
    head = batch_head % heads
    rows = q_block * BLOCK_TL + tl.arange(0, BLOCK_TL)
    dims = tl.arange(0, head_dim)
    valid = rows < tokens
    q = tl.load(
        q_ptr
        + batch * stride_b
        + rows[:, None].to(tl.int64) * stride_t
        + head * stride_h
        + dims[None, :],
        mask=valid[:, None],
        other=0.0,
    ).to(tl.float32)
    q_len = tl.minimum(BLOCK_TL, tokens - q_block * BLOCK_TL).to(tl.float32)
    centroid = tl.sum(q, axis=0) / q_len
    mean = tl.load(kc_mean_ptr + batch_head * head_dim + dims)
    variance = tl.load(kc_var_ptr + batch_head * head_dim + dims)
    log2_scale = scale * LOG2E_TL
    score_mean = tl.sum(centroid * mean, axis=0) * log2_scale
    score_var = tl.sum(centroid * centroid * variance, axis=0)
    score_var *= log2_scale * log2_scale
    tl.store(
        threshold_ptr + (batch * blocks + q_block) * heads + head,
        score_mean + tau * tl.sqrt(tl.maximum(score_var, 0.0) + 1.0e-6),
    )


@triton.autotune(
    configs=[
        triton.Config({"value_tile": 128, "group_size": 64}, num_warps=4, num_stages=1),
        triton.Config({"value_tile": 128, "group_size": 32}, num_warps=4, num_stages=1),
        triton.Config({"value_tile": 64, "group_size": 64}, num_warps=4, num_stages=1),
    ],
    key=["tokens", "max_blocks", "centroid_tail", "has_key_bias"],
)
@triton.jit
def _sol_attn_forward(
    q_ptr,
    kc_ptr,
    vc_ptr,
    output_ptr,
    qi_ptr,
    qs_ptr,
    ki_ptr,
    ks_ptr,
    vi_ptr,
    vs_ptr,
    threshold_ptr,
    key_bias_ptr,
    scale,
    sink_start,
    sink_end,
    sink_q_start,
    sink_q_end,
    max_blocks,
    tokens,
    q_padded,
    k_padded,
    blocks_padded,
    q_stride_b,
    q_stride_t,
    q_stride_h,
    heads: tl.constexpr,
    head_dim: tl.constexpr,
    blocks: tl.constexpr,
    value_tile: tl.constexpr,
    group_size: tl.constexpr,
    centroid_tail: tl.constexpr,
    has_key_bias: tl.constexpr,
):
    value_program = tl.program_id(0)
    q_block = tl.program_id(1)
    batch_head = tl.program_id(2)
    batch = batch_head // heads
    head = batch_head % heads

    group_offsets = tl.max_contiguous(tl.arange(0, group_size), group_size)
    token_offsets = tl.max_contiguous(tl.arange(0, BLOCK_TL), BLOCK_TL)
    dims = tl.arange(0, head_dim)
    value_dims = value_program * value_tile + tl.arange(0, value_tile)
    q_start = q_block * BLOCK_TL
    q_rows = q_start + token_offsets
    q_valid = q_rows < tokens

    q = tl.load(
        q_ptr
        + batch * q_stride_b
        + q_rows[:, None].to(tl.int64) * q_stride_t
        + head * q_stride_h
        + dims[None, :],
        mask=q_valid[:, None],
        other=0.0,
    )
    qi = tl.load(
        qi_ptr + ((batch * heads + head) * tokens + q_rows[:, None]) * head_dim + dims[None, :],
        mask=q_valid[:, None],
        other=0,
    )
    qs = tl.load(
        qs_ptr + (batch * heads + head) * q_padded + q_rows,
        mask=q_valid,
        other=0.0,
    )
    value_scale = tl.load(vs_ptr + batch_head * head_dim + value_dims)

    output = tl.zeros((BLOCK_TL, value_tile), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_TL,), dtype=tl.float32)
    row_max = tl.full((BLOCK_TL,), -float("inf"), dtype=tl.float32)
    log2_scale = scale * LOG2E_TL
    tail_length = tokens - (blocks - 1) * BLOCK_TL
    q_len = tl.minimum(BLOCK_TL, tokens - q_start).to(tl.float32)
    route_threshold = tl.load(
        threshold_ptr + (batch * blocks + q_block) * heads + head
    )
    q_in_sink = (q_block >= sink_q_start) & (q_block < sink_q_end)
    sink_count = tl.maximum(tl.minimum(sink_end, blocks) - sink_start, 0)
    non_sink_budget = tl.maximum(max_blocks - sink_count, 0)
    selected_non_sink = tl.zeros((), dtype=tl.int32)

    for group_start in range(0, blocks, group_size):
        block_indices = group_start + group_offsets
        valid_blocks = block_indices < blocks
        kc = tl.load(
            kc_ptr
            + ((batch * blocks_padded + block_indices[:, None]) * heads + head) * head_dim
            + dims[None, :]
        )
        vc = tl.load(
            vc_ptr
            + ((batch * blocks_padded + block_indices[:, None]) * heads + head) * head_dim
            + value_dims[None, :]
        )

        scores = tl.dot(q, kc.T).to(tl.float32) * log2_scale
        centroid_scores = tl.sum(scores, axis=0) / q_len
        sink_blocks = (
            (block_indices >= sink_start) & (block_indices < sink_end) & valid_blocks
        )
        routed = (
            (centroid_scores > route_threshold)
            | (tl.abs(q_block - block_indices) <= 1)
            | sink_blocks
        ) & valid_blocks
        candidates = tl.where(q_in_sink, valid_blocks, routed)
        non_sink = candidates & ~sink_blocks
        if max_blocks > 0:
            rank = selected_non_sink + tl.cumsum(non_sink.to(tl.int32), axis=0)
            non_sink = non_sink & (rank <= non_sink_budget)
            selected_non_sink += tl.sum(non_sink.to(tl.int32), axis=0)
        exact = sink_blocks | non_sink

        approximate = valid_blocks & ~exact
        if centroid_tail:
            tail_scores = tl.broadcast_to(
                centroid_scores[None, :], (BLOCK_TL, group_size)
            )
        else:
            tail_scores = scores
        tail_scores = tl.where(approximate[None, :], tail_scores, -float("inf"))
        tail_max = tl.max(tail_scores, axis=1)
        new_max = tl.maximum(row_max, tail_max)
        alpha = tl.math.exp2(tl.where(row_max == new_max, 0.0, row_max - new_max))
        tail_probability = tl.where(
            approximate[None, :],
            tl.math.exp2(tail_scores - new_max[:, None]),
            0.0,
        )
        output = output * alpha[:, None] + tl.dot(tail_probability.to(vc.dtype), vc)
        lengths = tl.where(block_indices == blocks - 1, tail_length, BLOCK_TL).to(
            tl.float32
        )
        row_sum = row_sum * alpha + tl.sum(tail_probability * lengths[None, :], axis=1)
        row_max = new_max

        exact_offsets = tl.where(exact, group_offsets, group_size)
        for _ in range(tl.sum(exact.to(tl.int32), axis=0)):
            offset = tl.min(exact_offsets, axis=0)
            exact_offsets = tl.where(group_offsets == offset, group_size, exact_offsets)
            key_block = group_start + offset
            key_rows = key_block * BLOCK_TL + token_offsets
            key_valid = key_rows < tokens
            ki = tl.load(
                ki_ptr
                + ((batch * heads + head) * tokens + key_rows[:, None]) * head_dim
                + dims[None, :],
                mask=key_valid[:, None],
                other=0,
            )
            ks = tl.load(
                ks_ptr + (batch * heads + head) * (k_padded // 16) + key_rows // 16,
                mask=key_valid,
                other=0.0,
            )
            score_i32 = tl.dot(qi, ki.T, out_dtype=tl.int32)
            exact_scores = score_i32.to(tl.float32) * (qs[:, None] * ks[None, :])
            exact_scores *= log2_scale
            if has_key_bias:
                bias = tl.load(
                    key_bias_ptr + batch * tokens + key_rows,
                    mask=key_valid,
                    other=-float("inf"),
                )
                exact_scores += bias[None, :]
            exact_scores = tl.where(key_valid[None, :], exact_scores, -float("inf"))

            block_max = tl.max(exact_scores, axis=1)
            new_max = tl.maximum(row_max, block_max)
            alpha = tl.math.exp2(row_max - new_max)
            probability = tl.math.exp2(exact_scores - new_max[:, None])
            row_sum = row_sum * alpha + tl.sum(probability, axis=1)

            vi = tl.load(
                vi_ptr
                + (batch_head * head_dim + value_dims[None, :]) * k_padded
                + key_rows[:, None],
                mask=key_valid[:, None],
                other=0,
            )
            p_scale = tl.maximum(tl.math.exp2(block_max - new_max), 1.0e-30) / 127.0
            pi = tl.minimum(probability / p_scale[:, None] + 0.5, 127.0).to(tl.int8)
            pv = tl.dot(pi, vi, out_dtype=tl.int32).to(tl.float32)
            pv *= p_scale[:, None] * value_scale[None, :]
            output = output * alpha[:, None] + pv
            row_max = new_max

    out_offsets = (
        ((batch * tokens + q_rows[:, None]) * heads + head) * head_dim
        + value_dims[None, :]
    )
    tl.store(
        output_ptr + out_offsets,
        (output / row_sum[:, None]).to(tl.bfloat16),
        mask=q_valid[:, None] & (value_dims < head_dim)[None, :],
    )


def _validate_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sink_blocks: list[int] | None,
    sink_q: list[int] | None,
) -> None:
    if q.dtype != torch.bfloat16:
        raise ValueError(f"sol_attn: q/k/v must be bfloat16, got {q.dtype}")
    check = sol_attn_common_call_rule(
        {"q": q, "k": k, "v": v, "sink_blocks": sink_blocks, "sink_q": sink_q}
    )
    if not check.success:
        raise ValueError(f"sol_attn: {check.failed_param}: {check.failure_reason}")
    if not q.is_cuda or torch.version.hip is None:
        raise ValueError("sol_attn: the HIP implementation requires a ROCm device")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.stride(-1) != 1:
            raise ValueError(f"sol_attn: {name} must have a contiguous last dim")
        if tensor.data_ptr() % 16:
            raise ValueError(f"sol_attn: {name} must be 16-byte aligned")
        for dim in range(3):
            if tensor.shape[dim] > 1 and tensor.stride(dim) % 8:
                raise ValueError(
                    f"sol_attn: {name} stride({dim}) must be a multiple of 8 elements"
                )


def _quantize_with_anchors(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    from comfy_kitchen.backends.eager.quantization import DTYPE_TO_CODE

    from . import _C, _dl, _sage_buffers, _stream

    qh, kh, vh = (tensor.permute(0, 2, 1, 3) for tensor in (q, k, v))
    buffers, anchor_indices = _sage_buffers(qh, kh, BLOCK)
    _C.sage_sdpa_quantize(
        _dl(qh),
        _dl(kh),
        _dl(vh),
        _dl(buffers["q_int8"]),
        _dl(buffers["q_scale"]),
        _dl(buffers["k_int8"]),
        _dl(buffers["k_scale"]),
        _dl(buffers["v_int8"]),
        _dl(buffers["v_scale"]),
        _dl(anchor_indices),
        BLOCK,
        DTYPE_TO_CODE[q.dtype],
        _stream(q),
    )
    return buffers, anchor_indices


def sol_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tau: float = 1.0,
    scale: float | None = None,
    sink_blocks: list[int] | None = None,
    sink_q: list[int] | None = None,
    max_blocks: int = 0,
    centroid_tail: bool = True,
    key_bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """INT8 Sol-Attn for ROCm over ``(B, T, H, 128)`` BF16 tensors."""
    _validate_inputs(q, k, v, sink_blocks, sink_q)
    batch, tokens, heads, head_dim = q.shape
    scale = head_dim**-0.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError(f"sol_attn: scale must be finite, got {scale}")
    if max_blocks < 0:
        raise ValueError(f"sol_attn: max_blocks must be non-negative, got {max_blocks}")

    blocks = triton.cdiv(tokens, BLOCK)
    blocks_padded = triton.cdiv(blocks, GROUP_PAD) * GROUP_PAD
    sb = [0, 0] if sink_blocks is None else list(sink_blocks)
    sq = [0, 0] if sink_q is None else list(sink_q)
    sink_count = max(0, min(sb[1], blocks) - sb[0])
    if max_blocks > 0 and sink_count > max_blocks:
        raise ValueError(
            f"sol_attn: max_blocks={max_blocks} is smaller than the "
            f"{sink_count}-block sink range {sb}"
        )

    normalized_bias = None
    if key_bias is not None:
        from comfy_kitchen.backends.eager.sol_attn import _normalize_key_bias

        normalized_bias = _normalize_key_bias(key_bias, batch, tokens, q.device)
        normalized_bias = (normalized_bias * LOG2E).expand(batch, tokens).contiguous()

    buffers, anchor_indices = _quantize_with_anchors(q, k, v)
    kc = torch.zeros(
        (batch, blocks_padded, heads, head_dim),
        device=q.device,
        dtype=torch.bfloat16,
    )
    vc = torch.zeros_like(kc)
    reduce_grid = (blocks, batch * heads)
    _reduce_kc_kernel[reduce_grid](
        k,
        kc,
        tokens,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        heads,
        blocks_padded,
        head_dim,
        num_warps=4,
        num_stages=2,
    )
    _reduce_vc_kernel[reduce_grid](
        v,
        vc,
        tokens,
        v.stride(0),
        v.stride(1),
        v.stride(2),
        heads,
        blocks_padded,
        head_dim,
        num_warps=4,
        num_stages=2,
    )
    _center_kc_kernel[reduce_grid](
        k,
        anchor_indices,
        kc,
        k.stride(0),
        k.stride(1),
        k.stride(2),
        heads,
        blocks_padded,
        head_dim,
        num_warps=4,
    )

    valid_kc = kc[:, :blocks].float().permute(0, 2, 1, 3)
    kc_mean = valid_kc.mean(dim=2).contiguous()
    kc_var = (valid_kc - kc_mean.unsqueeze(2)).square().mean(dim=2).contiguous()
    threshold = torch.empty((batch, blocks, heads), device=q.device, dtype=torch.float32)
    _threshold_kernel[(blocks, batch * heads)](
        q,
        kc_mean,
        kc_var,
        threshold,
        float(tau),
        scale,
        tokens,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        heads,
        blocks,
        head_dim,
        num_warps=4,
    )

    output = torch.empty_like(q, memory_format=torch.contiguous_format)
    q_padded = buffers["q_scale"].shape[-1]
    k_padded = buffers["v_int8"].shape[-1]
    def grid(meta):
        return (triton.cdiv(head_dim, meta["value_tile"]), blocks, batch * heads)

    _sol_attn_forward[grid](
        q,
        kc,
        vc,
        output,
        buffers["q_int8"],
        buffers["q_scale"],
        buffers["k_int8"],
        buffers["k_scale"],
        buffers["v_int8"],
        buffers["v_scale"],
        threshold,
        q if normalized_bias is None else normalized_bias,
        scale,
        int(sb[0]),
        int(sb[1]),
        int(sq[0]),
        int(sq[1]),
        int(max_blocks),
        tokens,
        q_padded,
        k_padded,
        blocks_padded,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        heads,
        head_dim,
        blocks,
        centroid_tail=bool(centroid_tail),
        has_key_bias=normalized_bias is not None,
    )
    return output


__all__ = ["sol_attn"]
