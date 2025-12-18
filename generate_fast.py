# unified_generate.py
# Drop-in helper: load Dream or LLaDA by name, then call model.generate(...) for both.

import types
import torch
from transformers import AutoTokenizer, AutoConfig

# Dream
from dream.modelling_dream import DreamModel
from dream.generation_utils_block import DreamGenerationMixin as DreamBlockMixin

# LLaDA
from llada.modelling_llada import LLaDAModelLM
from llada.generation import generate, generate_with_prefix_cache, generate_with_dual_cache

@torch.no_grad()
def _dream_generate(
    self,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    max_new_tokens: int = 256,
    steps: int = 8,
    block_length: int = 32, 
    algorithm: str = "confidence_threshold",
    threshold: float = 0.9,
    temperature: float = 0.0,
    top_p: float | None = None,
    top_k: int | None = None,
    dual_cache: bool = True,
    return_dict_in_generate: bool = False,
    **kwargs,
):
    """
    Unified .generate() for Dream.
    block_length: Used for cache blocking. For dual_cache, gen_length must be divisible by block_length.
    """
    out = self.diffusion_generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        output_history=False,
        return_dict_in_generate=True,
        steps=steps,
        block_length=block_length, 
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        alg=algorithm,
        threshold=threshold,
        dual_cache=dual_cache,
    )
    return out if return_dict_in_generate else out.sequences


@torch.no_grad()
def _llada_generate(
    self,
    input_ids: torch.Tensor,
    max_new_tokens: int = 256,
    steps: int = 8,
    block_length: int = 32,
    temperature: float = 0.0,
    remasking: str = "low_confidence",
    mask_id: int = 126336,
    threshold=None,
    parallel_factor=None,
    use_cache: bool = True,
    dual_cache: bool = True,
    **kwargs,
):
    """
    Unified .generate() for LLaDA.
    Here block_length is actually used by the underlying generate functions.
    
    Returns:
        out: Generated token IDs tensor of shape (B, L_prompt + gen_length)
        The full sequence including prompt is returned, just like generate_with_dual_cache
    """
    gen_length = max_new_tokens

    if use_cache:
        if dual_cache:
            out, _nfe = generate_with_dual_cache(
                self,
                input_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                remasking=remasking,
                mask_id=mask_id,
                threshold=threshold,
                factor=parallel_factor,
            )
        else:
            out, _nfe = generate_with_prefix_cache(
                self,
                input_ids,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=temperature,
                remasking=remasking,
                mask_id=mask_id,
                threshold=threshold,
                factor=parallel_factor,
            )
    else:
        out, _nfe = generate(
            self,
            input_ids,
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            mask_id=mask_id,
            threshold=threshold,
            factor=parallel_factor,
        )

    # Return just the tensor (full sequence), matching the official implementation
    # out is already a tensor of shape (B, L_prompt + gen_length)
    return out


def load_fast_diffusion_model_and_tokenizer(
    model_name_or_path: str,
    device: str = "cuda",
    dtype=torch.bfloat16,
):
    """
    Loads Dream or LLaDA based on name, patches model.generate(...) to a unified API,
    returns (model, tokenizer).
    """
    device_t = torch.device(device)
    name = model_name_or_path.lower()

    if "dream" in name:
        model = DreamModel.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).eval().to(device_t)

        # Use the block/cached mixin (supports dual_cache etc.)
        model.diffusion_generate = types.MethodType(DreamBlockMixin.diffusion_generate, model)
        model._sample = types.MethodType(DreamBlockMixin._sample, model)

        # Patch unified .generate()
        model.generate = types.MethodType(_dream_generate, model)

    elif "llada" in name:
        # Match the official eval_llada.py approach
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.flash_attention = True
        
        model = LLaDAModelLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            torch_dtype=dtype,
            config=config,
        ).eval().to(device_t)

        # Patch unified .generate()
        model.generate = types.MethodType(_llada_generate, model)

    else:
        raise ValueError(
            f"Can't infer model family from '{model_name_or_path}'. "
            f"Expected name/path containing 'dream' or 'llada'."
        )

    tok = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    return model, tok

def build_prompt(tokenizer, model_path: str, question: str) -> str:
    is_instruct = "instruct" in model_path.lower()
    if not is_instruct:
        return question
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    
def main():
    pass

if __name__ == "__main__":
    main()