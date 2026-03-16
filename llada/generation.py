# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

from __future__ import annotations

import inspect
from collections.abc import Sequence
from typing import Optional

import torch


def _normalize_token_id_sequence(token_ids: Optional[Sequence[int]]) -> tuple[int, ...]:
    if token_ids is None:
        return ()

    normalized: list[int] = []
    seen: set[int] = set()
    for token_id in token_ids:
        if token_id is None:
            continue
        token_id = int(token_id)
        if token_id not in seen:
            seen.add(token_id)
            normalized.append(token_id)
    return tuple(normalized)


def _resolve_suppressed_token_ids(
    suppressed_sample_token_ids: Optional[Sequence[int]] = None,
    suppressed_confidence_token_ids: Optional[Sequence[int]] = None,
    eos_token_id: Optional[int] = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    sample_ids = list(_normalize_token_id_sequence(suppressed_sample_token_ids))
    confidence_ids = list(_normalize_token_id_sequence(suppressed_confidence_token_ids))

    if eos_token_id is not None and logits_eos_inf:
        sample_ids.append(int(eos_token_id))
    if eos_token_id is not None and confidence_eos_eot_inf:
        confidence_ids.append(int(eos_token_id))

    return (
        _normalize_token_id_sequence(sample_ids),
        _normalize_token_id_sequence(confidence_ids),
    )


def _suppress_logits(
    logits: torch.Tensor,
    suppressed_token_ids: Sequence[int],
) -> torch.Tensor:
    if not suppressed_token_ids:
        return logits

    suppressed = logits.clone()
    token_index = torch.tensor(
        suppressed_token_ids,
        device=logits.device,
        dtype=torch.long,
    )
    suppressed.index_fill_(-1, token_index, torch.finfo(suppressed.dtype).min)
    return suppressed


def sample_gumbel_argmax(
    logits: torch.Tensor,
    temperature: float,
    chunk_size: Optional[int] = 4096,
    gumbel_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Sample token ids with Gumbel-Max without materializing a full noisy logits tensor.

    This is distribution-equivalent to taking `argmax(logits + temperature * gumbel)`
    while reducing peak memory from O(B * L * V) extra storage to O(B * L * chunk_size).
    When `temperature == 0`, this is exactly greedy argmax and remains deterministic.
    """
    if temperature == 0:
        return torch.argmax(logits, dim=-1)

    if chunk_size is None or chunk_size <= 0:
        chunk_size = logits.shape[-1]

    batch_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    best_scores = torch.full(
        batch_shape,
        torch.finfo(gumbel_dtype).min,
        device=logits.device,
        dtype=gumbel_dtype,
    )
    best_indices = torch.zeros(batch_shape, device=logits.device, dtype=torch.long)
    tiny = torch.finfo(gumbel_dtype).tiny

    for start in range(0, vocab_size, chunk_size):
        stop = min(start + chunk_size, vocab_size)
        logits_chunk = logits[..., start:stop].to(gumbel_dtype)
        uniform = torch.rand(logits_chunk.shape, device=logits.device, dtype=gumbel_dtype)
        uniform.clamp_(min=tiny, max=1.0 - torch.finfo(gumbel_dtype).eps)
        chunk_scores = logits_chunk - temperature * torch.log(-torch.log(uniform))
        chunk_best_scores, chunk_best_idx = torch.max(chunk_scores, dim=-1)

        update_mask = chunk_best_scores > best_scores
        best_scores = torch.where(update_mask, chunk_best_scores, best_scores)
        best_indices = torch.where(update_mask, chunk_best_idx + start, best_indices)

    return best_indices


def get_num_transfer_tokens(block_mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Distribute masked-token transfers evenly across diffusion steps."""
    if block_mask_index.ndim != 2:
        raise ValueError(f"Expected block_mask_index to have shape (B, L), got {tuple(block_mask_index.shape)}")
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")

    total = block_mask_index.sum(dim=1, dtype=torch.long)
    base = torch.div(total, steps, rounding_mode="floor")
    remainder = total - base * steps

    num_transfer_tokens = base.unsqueeze(1).expand(-1, steps).clone()
    cols = torch.arange(steps, device=block_mask_index.device).unsqueeze(0)
    num_transfer_tokens += (cols < remainder.unsqueeze(1)).to(torch.long)
    return num_transfer_tokens


def _prepare_generation_attention_mask(
    prompt: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    gen_length: int,
) -> Optional[torch.Tensor]:
    if attention_mask is None:
        return None

    gen_attention_mask = torch.ones(
        (prompt.shape[0], gen_length),
        dtype=attention_mask.dtype,
        device=prompt.device,
    )
    return torch.cat([attention_mask.to(prompt.device), gen_attention_mask], dim=-1)


def _compute_token_confidence(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    logits_fp = logits.to(compute_dtype)
    chosen_logits = torch.gather(logits_fp, dim=-1, index=token_ids.unsqueeze(-1)).squeeze(-1)
    log_denom = torch.logsumexp(logits_fp, dim=-1)
    return torch.exp(chosen_logits - log_denom)


def _propose_tokens_and_confidence(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    suppressed_sample_token_ids: Sequence[int],
    suppressed_confidence_token_ids: Sequence[int],
    gumbel_chunk_size: Optional[int],
    gumbel_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    sample_logits = _suppress_logits(logits, suppressed_sample_token_ids)
    proposed_tokens = sample_gumbel_argmax(
        sample_logits,
        temperature=temperature,
        chunk_size=gumbel_chunk_size,
        gumbel_dtype=gumbel_dtype,
    )

    if remasking == "low_confidence":
        confidence_logits = sample_logits
        extra_confidence_ids = tuple(
            token_id for token_id in suppressed_confidence_token_ids if token_id not in suppressed_sample_token_ids
        )
        if extra_confidence_ids:
            confidence_logits = _suppress_logits(confidence_logits, extra_confidence_ids)
        token_confidence = _compute_token_confidence(confidence_logits, proposed_tokens)
    elif remasking == "random":
        token_confidence = torch.rand(mask_index.shape, device=mask_index.device, dtype=torch.float32)
    else:
        raise NotImplementedError(remasking)

    proposed_tokens = torch.where(mask_index, proposed_tokens, x)
    neg_inf = torch.full_like(token_confidence, torch.finfo(token_confidence.dtype).min)
    confidence = torch.where(mask_index, token_confidence, neg_inf)
    return proposed_tokens, confidence


def get_transfer_index(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: Optional[torch.Tensor],
    threshold: Optional[float] = None,
    suppressed_sample_token_ids: Optional[Sequence[int]] = None,
    suppressed_confidence_token_ids: Optional[Sequence[int]] = None,
    *,
    eos_token_id: Optional[int] = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    gumbel_chunk_size: Optional[int] = 4096,
    gumbel_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return proposed tokens and the positions to transfer this step."""
    if mask_index.shape != x.shape or logits.shape[:2] != x.shape:
        raise ValueError(
            "Expected logits.shape[:2], mask_index.shape, and x.shape to match; "
            f"got logits={tuple(logits.shape)}, mask_index={tuple(mask_index.shape)}, x={tuple(x.shape)}"
        )

    sample_ids, confidence_ids = _resolve_suppressed_token_ids(
        suppressed_sample_token_ids=suppressed_sample_token_ids,
        suppressed_confidence_token_ids=suppressed_confidence_token_ids,
        eos_token_id=eos_token_id,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
    )

    proposed_tokens, confidence = _propose_tokens_and_confidence(
        logits=logits,
        temperature=temperature,
        remasking=remasking,
        mask_index=mask_index,
        x=x,
        suppressed_sample_token_ids=sample_ids,
        suppressed_confidence_token_ids=confidence_ids,
        gumbel_chunk_size=gumbel_chunk_size,
        gumbel_dtype=gumbel_dtype,
    )

    if threshold is not None:
        transfer_index = mask_index & (confidence >= threshold)
        row_has_mask = mask_index.any(dim=1)
        if row_has_mask.any():
            best_indices = torch.argmax(confidence, dim=1, keepdim=True)
            forced_transfer = torch.zeros_like(transfer_index).scatter_(1, best_indices, True)
            transfer_index |= forced_transfer & row_has_mask.unsqueeze(1)
        return proposed_tokens, transfer_index & mask_index

    if num_transfer_tokens is None:
        raise ValueError("num_transfer_tokens must be provided when threshold is None")

    if num_transfer_tokens.ndim == 2 and num_transfer_tokens.shape[1] == 1:
        num_transfer_tokens = num_transfer_tokens.squeeze(1)
    if num_transfer_tokens.ndim != 1 or num_transfer_tokens.shape[0] != x.shape[0]:
        raise ValueError(
            f"Expected num_transfer_tokens to have shape (B,), got {tuple(num_transfer_tokens.shape)}"
        )

    mask_count = mask_index.sum(dim=1, dtype=torch.long)
    quota = num_transfer_tokens.to(device=x.device, dtype=torch.long).clamp(min=0)
    quota = torch.minimum(quota, mask_count)
    max_k = int(quota.max().item())

    transfer_index = torch.zeros_like(mask_index)
    if max_k == 0:
        return proposed_tokens, transfer_index

    topk_indices = torch.topk(confidence, k=max_k, dim=1, largest=True, sorted=True).indices
    select = torch.arange(max_k, device=x.device).unsqueeze(0) < quota.unsqueeze(1)
    transfer_index.scatter_(1, topk_indices, select)
    return proposed_tokens, transfer_index & mask_index


def get_transfer_index_dynamic(
    logits: torch.Tensor,
    temperature: float,
    remasking: str,
    mask_index: torch.Tensor,
    x: torch.Tensor,
    num_transfer_tokens: Optional[torch.Tensor],
    factor: float = 1.0,
    suppressed_sample_token_ids: Optional[Sequence[int]] = None,
    suppressed_confidence_token_ids: Optional[Sequence[int]] = None,
    *,
    eos_token_id: Optional[int] = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    gumbel_chunk_size: Optional[int] = 4096,
    gumbel_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic quota selection from the public LLaDA-style confidence schedule."""
    del num_transfer_tokens

    if factor <= 0:
        raise ValueError(f"factor must be positive, got {factor}")

    sample_ids, confidence_ids = _resolve_suppressed_token_ids(
        suppressed_sample_token_ids=suppressed_sample_token_ids,
        suppressed_confidence_token_ids=suppressed_confidence_token_ids,
        eos_token_id=eos_token_id,
        logits_eos_inf=logits_eos_inf,
        confidence_eos_eot_inf=confidence_eos_eot_inf,
    )

    proposed_tokens, confidence = _propose_tokens_and_confidence(
        logits=logits,
        temperature=temperature,
        remasking=remasking,
        mask_index=mask_index,
        x=x,
        suppressed_sample_token_ids=sample_ids,
        suppressed_confidence_token_ids=confidence_ids,
        gumbel_chunk_size=gumbel_chunk_size,
        gumbel_dtype=gumbel_dtype,
    )

    transfer_index = torch.zeros_like(mask_index)
    for row_idx in range(mask_index.shape[0]):
        row_mask = mask_index[row_idx]
        num_masked = int(row_mask.sum().item())
        if num_masked == 0:
            continue

        row_confidence = torch.sort(confidence[row_idx][row_mask], descending=True).values
        ranks = torch.arange(1, num_masked + 1, device=confidence.device, dtype=row_confidence.dtype)
        thresholds = 1.0 - (factor / (ranks + 1.0))
        thresholds[0] = torch.finfo(row_confidence.dtype).min
        valid = row_confidence >= thresholds
        transfer_count = int(valid.sum().item())
        transfer_count = max(1, min(transfer_count, num_masked))

        selected = torch.topk(confidence[row_idx], k=transfer_count, largest=True, sorted=True).indices
        transfer_index[row_idx, selected] = True

    return proposed_tokens, transfer_index & mask_index


def _apply_updates_(x: torch.Tensor, updates: torch.Tensor, transfer_index: torch.Tensor) -> None:
    x.copy_(torch.where(transfer_index, updates, x))


def _truncate_past_key_values(
    past_key_values,
    prefix_length: int,
):
    return tuple(
        tuple(cache[:, :, :prefix_length, :] for cache in layer_past)
        for layer_past in past_key_values
    )


def _model_supports_replace_position(model) -> bool:
    try:
        signature = inspect.signature(model.forward)
    except (TypeError, ValueError):
        return False
    return "replace_position" in signature.parameters


def generate(
    model,
    prompt,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
    suppressed_sample_token_ids: Optional[Sequence[int]] = None,
    suppressed_confidence_token_ids: Optional[Sequence[int]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    eos_token_id: Optional[int] = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    gumbel_chunk_size: Optional[int] = 4096,
    gumbel_dtype: torch.dtype = torch.float64,
):
    device = model.device
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, : prompt.shape[1]] = prompt.to(device)
    generation_attention_mask = _prepare_generation_attention_mask(prompt, attention_mask, gen_length)

    if gen_length % block_length != 0:
        raise ValueError(f"gen_length={gen_length} must be divisible by block_length={block_length}")
    num_blocks = gen_length // block_length

    if steps % num_blocks != 0:
        raise ValueError(f"steps={steps} must be divisible by num_blocks={num_blocks}")
    steps_per_block = steps // num_blocks

    nfe = 0
    prompt_length = prompt.shape[1]
    for num_block in range(num_blocks):
        start = prompt_length + num_block * block_length
        end = start + block_length
        block_mask_index = x[:, start:end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        step_idx = 0
        while (x[:, start:end] == mask_id).any():
            mask_index = x == mask_id
            mask_index[:, end:] = False
            logits = model(x, attention_mask=generation_attention_mask).logits
            nfe += 1

            if factor is None:
                proposed_tokens, transfer_index = get_transfer_index(
                    logits=logits,
                    temperature=temperature,
                    remasking=remasking,
                    mask_index=mask_index,
                    x=x,
                    num_transfer_tokens=num_transfer_tokens[:, step_idx] if threshold is None else None,
                    threshold=threshold,
                    suppressed_sample_token_ids=suppressed_sample_token_ids,
                    suppressed_confidence_token_ids=suppressed_confidence_token_ids,
                    eos_token_id=eos_token_id,
                    logits_eos_inf=logits_eos_inf,
                    confidence_eos_eot_inf=confidence_eos_eot_inf,
                    gumbel_chunk_size=gumbel_chunk_size,
                    gumbel_dtype=gumbel_dtype,
                )
            else:
                proposed_tokens, transfer_index = get_transfer_index_dynamic(
                    logits=logits,
                    temperature=temperature,
                    remasking=remasking,
                    mask_index=mask_index,
                    x=x,
                    num_transfer_tokens=None,
                    factor=factor,
                    suppressed_sample_token_ids=suppressed_sample_token_ids,
                    suppressed_confidence_token_ids=suppressed_confidence_token_ids,
                    eos_token_id=eos_token_id,
                    logits_eos_inf=logits_eos_inf,
                    confidence_eos_eot_inf=confidence_eos_eot_inf,
                    gumbel_chunk_size=gumbel_chunk_size,
                    gumbel_dtype=gumbel_dtype,
                )

            _apply_updates_(x, proposed_tokens, transfer_index)
            step_idx += 1

    return x, nfe


def generate_with_prefix_cache(
    model,
    prompt,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
    suppressed_sample_token_ids: Optional[Sequence[int]] = None,
    suppressed_confidence_token_ids: Optional[Sequence[int]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    eos_token_id: Optional[int] = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    gumbel_chunk_size: Optional[int] = 4096,
    gumbel_dtype: torch.dtype = torch.float64,
):
    device = model.device
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, : prompt.shape[1]] = prompt.to(device)
    generation_attention_mask = _prepare_generation_attention_mask(prompt, attention_mask, gen_length)

    if gen_length % block_length != 0:
        raise ValueError(f"gen_length={gen_length} must be divisible by block_length={block_length}")
    num_blocks = gen_length // block_length

    if steps % num_blocks != 0:
        raise ValueError(f"steps={steps} must be divisible by num_blocks={num_blocks}")
    steps_per_block = steps // num_blocks

    nfe = 0
    prompt_length = prompt.shape[1]

    for num_block in range(num_blocks):
        start = prompt_length + num_block * block_length
        end = start + block_length

        block_mask_index = x[:, start:end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        output = model(x, use_cache=True, attention_mask=generation_attention_mask)
        past_key_values = _truncate_past_key_values(output.past_key_values, start)
        nfe += 1

        mask_index = x == mask_id
        mask_index[:, end:] = False
        if factor is None:
            proposed_tokens, transfer_index = get_transfer_index(
                logits=output.logits,
                temperature=temperature,
                remasking=remasking,
                mask_index=mask_index,
                x=x,
                num_transfer_tokens=num_transfer_tokens[:, 0] if threshold is None else None,
                threshold=threshold,
                suppressed_sample_token_ids=suppressed_sample_token_ids,
                suppressed_confidence_token_ids=suppressed_confidence_token_ids,
                eos_token_id=eos_token_id,
                logits_eos_inf=logits_eos_inf,
                confidence_eos_eot_inf=confidence_eos_eot_inf,
                gumbel_chunk_size=gumbel_chunk_size,
                gumbel_dtype=gumbel_dtype,
            )
        else:
            proposed_tokens, transfer_index = get_transfer_index_dynamic(
                logits=output.logits,
                temperature=temperature,
                remasking=remasking,
                mask_index=mask_index,
                x=x,
                num_transfer_tokens=None,
                factor=factor,
                suppressed_sample_token_ids=suppressed_sample_token_ids,
                suppressed_confidence_token_ids=suppressed_confidence_token_ids,
                eos_token_id=eos_token_id,
                logits_eos_inf=logits_eos_inf,
                confidence_eos_eot_inf=confidence_eos_eot_inf,
                gumbel_chunk_size=gumbel_chunk_size,
                gumbel_dtype=gumbel_dtype,
            )
        _apply_updates_(x, proposed_tokens, transfer_index)

        step_idx = 1
        while (x[:, start:end] == mask_id).any():
            suffix = x[:, start:]
            suffix_mask = suffix == mask_id
            suffix_mask[:, block_length:] = False

            output = model(
                suffix,
                past_key_values=past_key_values,
                use_cache=True,
                attention_mask=generation_attention_mask,
            )
            nfe += 1

            if factor is None:
                proposed_tokens, transfer_index = get_transfer_index(
                    logits=output.logits,
                    temperature=temperature,
                    remasking=remasking,
                    mask_index=suffix_mask,
                    x=suffix,
                    num_transfer_tokens=num_transfer_tokens[:, step_idx] if threshold is None else None,
                    threshold=threshold,
                    suppressed_sample_token_ids=suppressed_sample_token_ids,
                    suppressed_confidence_token_ids=suppressed_confidence_token_ids,
                    eos_token_id=eos_token_id,
                    logits_eos_inf=logits_eos_inf,
                    confidence_eos_eot_inf=confidence_eos_eot_inf,
                    gumbel_chunk_size=gumbel_chunk_size,
                    gumbel_dtype=gumbel_dtype,
                )
            else:
                proposed_tokens, transfer_index = get_transfer_index_dynamic(
                    logits=output.logits,
                    temperature=temperature,
                    remasking=remasking,
                    mask_index=suffix_mask,
                    x=suffix,
                    num_transfer_tokens=None,
                    factor=factor,
                    suppressed_sample_token_ids=suppressed_sample_token_ids,
                    suppressed_confidence_token_ids=suppressed_confidence_token_ids,
                    eos_token_id=eos_token_id,
                    logits_eos_inf=logits_eos_inf,
                    confidence_eos_eot_inf=confidence_eos_eot_inf,
                    gumbel_chunk_size=gumbel_chunk_size,
                    gumbel_dtype=gumbel_dtype,
                )

            x[:, start:] = torch.where(transfer_index, proposed_tokens, x[:, start:])
            step_idx += 1

    return x, nfe


def generate_with_dual_cache(
    model,
    prompt,
    steps: int = 128,
    gen_length: int = 128,
    block_length: int = 128,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold: Optional[float] = None,
    factor: Optional[float] = None,
    suppressed_sample_token_ids: Optional[Sequence[int]] = None,
    suppressed_confidence_token_ids: Optional[Sequence[int]] = None,
    attention_mask: Optional[torch.Tensor] = None,
    *,
    eos_token_id: Optional[int] = None,
    logits_eos_inf: bool = False,
    confidence_eos_eot_inf: bool = False,
    gumbel_chunk_size: Optional[int] = 4096,
    gumbel_dtype: torch.dtype = torch.float64,
):
    """Custom-fork dual-cache path.

    Public upstream `modeling_llada.py` does not expose `replace_position`, so we
    only use this path when the loaded model explicitly supports that argument.
    Otherwise we fall back to the correctness-preserving prefix-cache path.
    """
    if not _model_supports_replace_position(model):
        return generate_with_prefix_cache(
            model=model,
            prompt=prompt,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            mask_id=mask_id,
            threshold=threshold,
            factor=factor,
            suppressed_sample_token_ids=suppressed_sample_token_ids,
            suppressed_confidence_token_ids=suppressed_confidence_token_ids,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            logits_eos_inf=logits_eos_inf,
            confidence_eos_eot_inf=confidence_eos_eot_inf,
            gumbel_chunk_size=gumbel_chunk_size,
            gumbel_dtype=gumbel_dtype,
        )

    device = model.device
    x = torch.full(
        (prompt.shape[0], prompt.shape[1] + gen_length),
        mask_id,
        dtype=torch.long,
        device=device,
    )
    x[:, : prompt.shape[1]] = prompt.to(device)
    generation_attention_mask = _prepare_generation_attention_mask(prompt, attention_mask, gen_length)

    if gen_length % block_length != 0:
        raise ValueError(f"gen_length={gen_length} must be divisible by block_length={block_length}")
    num_blocks = gen_length // block_length

    if steps % num_blocks != 0:
        raise ValueError(f"steps={steps} must be divisible by num_blocks={num_blocks}")
    steps_per_block = steps // num_blocks

    nfe = 0
    prompt_length = prompt.shape[1]

    for num_block in range(num_blocks):
        start = prompt_length + num_block * block_length
        end = start + block_length
        block_mask_index = x[:, start:end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        output = model(x, use_cache=True, attention_mask=generation_attention_mask)
        past_key_values = output.past_key_values
        nfe += 1

        mask_index = x == mask_id
        mask_index[:, end:] = False
        if factor is None:
            proposed_tokens, transfer_index = get_transfer_index(
                logits=output.logits,
                temperature=temperature,
                remasking=remasking,
                mask_index=mask_index,
                x=x,
                num_transfer_tokens=num_transfer_tokens[:, 0] if threshold is None else None,
                threshold=threshold,
                suppressed_sample_token_ids=suppressed_sample_token_ids,
                suppressed_confidence_token_ids=suppressed_confidence_token_ids,
                eos_token_id=eos_token_id,
                logits_eos_inf=logits_eos_inf,
                confidence_eos_eot_inf=confidence_eos_eot_inf,
                gumbel_chunk_size=gumbel_chunk_size,
                gumbel_dtype=gumbel_dtype,
            )
        else:
            proposed_tokens, transfer_index = get_transfer_index_dynamic(
                logits=output.logits,
                temperature=temperature,
                remasking=remasking,
                mask_index=mask_index,
                x=x,
                num_transfer_tokens=None,
                factor=factor,
                suppressed_sample_token_ids=suppressed_sample_token_ids,
                suppressed_confidence_token_ids=suppressed_confidence_token_ids,
                eos_token_id=eos_token_id,
                logits_eos_inf=logits_eos_inf,
                confidence_eos_eot_inf=confidence_eos_eot_inf,
                gumbel_chunk_size=gumbel_chunk_size,
                gumbel_dtype=gumbel_dtype,
            )
        _apply_updates_(x, proposed_tokens, transfer_index)

        replace_position = torch.zeros_like(x, dtype=torch.bool)
        replace_position[:, start:end] = True

        step_idx = 1
        while (x[:, start:end] == mask_id).any():
            output = model(
                x[:, start:end],
                past_key_values=past_key_values,
                use_cache=True,
                replace_position=replace_position,
                attention_mask=generation_attention_mask,
            )
            past_key_values = output.past_key_values
            nfe += 1

            block_mask = x[:, start:end] == mask_id
            if factor is None:
                proposed_tokens, transfer_index = get_transfer_index(
                    logits=output.logits,
                    temperature=temperature,
                    remasking=remasking,
                    mask_index=block_mask,
                    x=x[:, start:end],
                    num_transfer_tokens=num_transfer_tokens[:, step_idx] if threshold is None else None,
                    threshold=threshold,
                    suppressed_sample_token_ids=suppressed_sample_token_ids,
                    suppressed_confidence_token_ids=suppressed_confidence_token_ids,
                    eos_token_id=eos_token_id,
                    logits_eos_inf=logits_eos_inf,
                    confidence_eos_eot_inf=confidence_eos_eot_inf,
                    gumbel_chunk_size=gumbel_chunk_size,
                    gumbel_dtype=gumbel_dtype,
                )
            else:
                proposed_tokens, transfer_index = get_transfer_index_dynamic(
                    logits=output.logits,
                    temperature=temperature,
                    remasking=remasking,
                    mask_index=block_mask,
                    x=x[:, start:end],
                    num_transfer_tokens=None,
                    factor=factor,
                    suppressed_sample_token_ids=suppressed_sample_token_ids,
                    suppressed_confidence_token_ids=suppressed_confidence_token_ids,
                    eos_token_id=eos_token_id,
                    logits_eos_inf=logits_eos_inf,
                    confidence_eos_eot_inf=confidence_eos_eot_inf,
                    gumbel_chunk_size=gumbel_chunk_size,
                    gumbel_dtype=gumbel_dtype,
                )

            x[:, start:end] = torch.where(transfer_index, proposed_tokens, x[:, start:end])
            step_idx += 1

    return x, nfe
