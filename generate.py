import types
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Optional


def add_gumbel_noise(logits, temperature):
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64)
        + base
    )
    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


class OfficialLLaDAGenerator:
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        num_diffusion_steps: int = 128,
        steps: int = None,  # Alias for num_diffusion_steps
        block_length: int = 32,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = "low_confidence",
        mask_id: int = 156895,
        logits_eos_inf: bool = False,
        confidence_eos_eot_inf: bool = False,
        **kwargs
    ) -> torch.Tensor:

        # Handle both 'steps' and 'num_diffusion_steps' parameter names
        if steps is not None:
            num_diffusion_steps = steps

        # Use model_path from self (set during initialization)
        if "llada" in self.model_path.lower():
            mask_id = 126336
        if "llada-moe" in self.model_path.lower():
            mask_id = 126336

        prompt = input_ids
        gen_length = max_new_tokens
        diffusion_steps = num_diffusion_steps

        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long,
        ).to(self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        if attention_mask is not None:
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (prompt.shape[0], gen_length),
                        dtype=attention_mask.dtype,
                        device=self.model.device,
                    ),
                ],
                dim=-1,
            )

        prompt_index = (x != mask_id)
        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length
        assert diffusion_steps % num_blocks == 0
        steps_per_block = diffusion_steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (
                x[
                    :,
                    prompt.shape[1]
                    + num_block * block_length : prompt.shape[1]
                    + (num_block + 1) * block_length,
                ]
                == mask_id
            )
            num_transfer_tokens = get_num_transfer_tokens(
                block_mask_index, steps_per_block
            )

            for i in range(steps_per_block):
                mask_index = (x == mask_id)

                # Classifier-free guidance
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)

                    if attention_mask is not None:
                        attention_mask_ = torch.cat(
                            [attention_mask, attention_mask], dim=0
                        )
                        logits = self.model(x_, attention_mask=attention_mask_).logits
                    else:
                        logits = self.model(x_).logits

                    logits, un_logits = torch.chunk(logits, 2, dim=0)
                    logits = un_logits + (cfg_scale + 1) * (logits - un_logits)

                else:
                    if attention_mask is not None:
                        logits = self.model(x, attention_mask=attention_mask).logits
                    else:
                        logits = self.model(x).logits

                # optionally remove EOS token from consideration
                if logits_eos_inf:
                    try:
                        logits[:, :, self.eos_token_id] = -float("inf")
                    except Exception:
                        pass

                logits_with_noise = add_gumbel_noise(
                    logits, temperature=temperature
                )
                x0 = torch.argmax(logits_with_noise, dim=-1)

                if confidence_eos_eot_inf:
                    try:
                        logits_with_noise[
                            :, :, self.eos_token_id
                        ] = logits[:, :, self.eos_token_id] = -float("inf")
                    except Exception:
                        pass

                if remasking == "low_confidence":
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )
                elif remasking == "random":
                    x0_p = torch.rand_like(x0_p)
                else:
                    raise NotImplementedError(remasking)

                # fall-off region beyond this block is made invalid
                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length :] = -float(
                    "inf"
                )

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -float("inf"))

                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(
                        confidence[j], k=num_transfer_tokens[j, i]
                    )
                    transfer_index[j, select_index] = True

                x[transfer_index] = x0[transfer_index]

        return x


class OfficialDreamGenerator:
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        num_diffusion_steps: int = 64,
        temperature: float = 0.2,
        alg: str = "confidence_threshold",
        alg_temp: float = 0.0,
        top_p: float | None = None,
        top_k: int | None = None,
        **kwargs
    ) -> torch.Tensor:
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "steps": num_diffusion_steps,
            "temperature": temperature,
            "alg": alg,
            "alg_temp": alg_temp,
        }
        if attention_mask is not None:
            gen_kwargs["attention_mask"] = attention_mask
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
        if top_k is not None:
            gen_kwargs["top_k"] = top_k

        # Uses Dream’s official diffusion_generate
        return self.model.diffusion_generate(input_ids, **gen_kwargs)


def load_diffusion_model_and_tokenizer(
    model_name_or_path: str, device: str = "cuda"
):
    device_t = torch.device(device)
    name = model_name_or_path.lower()

    model = AutoModel.from_pretrained(
        model_name_or_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device_t).eval()

    tok = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    # Attach generator
    if "dream" in name:
        gen_wrapper = OfficialDreamGenerator()
        gen_wrapper.model = model
        gen_wrapper.model_path = model_name_or_path
        # Use lambda to avoid double-binding issues
        model.generate = lambda **kwargs: gen_wrapper.generate(**kwargs)
    elif "llada" in name:
        gen_wrapper = OfficialLLaDAGenerator()
        gen_wrapper.model = model
        gen_wrapper.model_path = model_name_or_path
        # Use lambda to avoid double-binding issues
        model.generate = lambda **kwargs: gen_wrapper.generate(**kwargs)
    else:
        raise ValueError(f"Cannot deduce model family from '{model_name_or_path}'")

    return model, tok

def build_prompt(tokenizer, model_path: str, question: str) -> str:
    is_instruct = "instruct" in model_path.lower()
    if not is_instruct:
        return question
    messages = [{"role": "user", "content": question}]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
