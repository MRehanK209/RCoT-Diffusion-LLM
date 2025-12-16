"""
Diffusion-based text generation using official LLaDA implementation.

Based on https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
"""

import torch
import numpy as np
import torch.nn.functional as F
from typing import Optional
from abc import ABC, abstractmethod


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (- torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


class DiffusionModelBase(ABC):
    """Base class for diffusion language models."""
    
    def __init__(self, model, tokenizer, model_path: str):
        """
        Initialize the diffusion model.
        
        Args:
            model: The underlying HuggingFace model
            tokenizer: The tokenizer
            model_path: Path or identifier of the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_path = model_path
        self.model.eval()
    
    @abstractmethod
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.0,
        **kwargs
    ) -> torch.Tensor:
        """Generate text using the model."""
        pass
    
    @property
    def device(self):
        """Get the model's device."""
        return self.model.device


class LLaDAModel(DiffusionModelBase):
    """
    LLaDA model using official implementation from:
    https://github.com/ML-GSAI/LLaDA/blob/main/generate.py
    """
    
    # Valid parameters for LLaDA
    VALID_PARAMS = {
        'input_ids', 'attention_mask', 'max_new_tokens', 'num_diffusion_steps',
        'block_length', 'temperature', 'cfg_scale', 'remasking', 'mask_id',
        'logits_eos_inf', 'confidence_eos_eot_inf'
    }
    
    # Parameters that are NOT supported by LLaDA (Dream-specific)
    INVALID_PARAMS = {'alg', 'alg_temp', 'steps', 'top_p', 'top_k'}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        num_diffusion_steps: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = 'low_confidence',
        mask_id: int = 126336,
        logits_eos_inf: bool = False,
        confidence_eos_eot_inf: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate using LLaDA's diffusion sampling process.
        
        Args:
            input_ids: Input token IDs (prompt)
            attention_mask: Attention mask for the input
            max_new_tokens: Number of tokens to generate
            num_diffusion_steps: Sampling steps, less than or equal to max_new_tokens
            block_length: Block length, less than or equal to max_new_tokens
            temperature: Categorical distribution sampling temperature
            cfg_scale: Unsupervised classifier-free guidance scale
            remasking: Remasking strategy ('low_confidence' or 'random')
            mask_id: The token id of [MASK] is 126336
            logits_eos_inf: Whether to set the logits of EOS token to -inf
            confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf
            **kwargs: Additional arguments (will be validated)
        """
        # Validate and warn about invalid parameters
        if kwargs:
            invalid_params = set(kwargs.keys()) & self.INVALID_PARAMS
            if invalid_params:
                print(f"WARNING: LLaDA does not support these parameters (ignoring): {invalid_params}")
            
            # Warn about any unrecognized parameters
            unrecognized = set(kwargs.keys()) - self.VALID_PARAMS - self.INVALID_PARAMS
            if unrecognized:
                print(f"WARNING: Unrecognized parameters for LLaDA (ignoring): {unrecognized}")
        
        prompt = input_ids
        gen_length = max_new_tokens
        steps = num_diffusion_steps
        
        # Initialize full sequence with mask tokens
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long
        ).to(self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        # Extend attention mask if provided
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=self.model.device)
            ], dim=-1)

        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (
                x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length]
                == mask_id
            )
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            
            for i in range(steps_per_block):
                mask_index = (x == mask_id)
                
                # Classifier-free guidance
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    
                    if attention_mask is not None:
                        attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
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

                if logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
                
                if confidence_eos_eot_inf:
                    logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        return x


class LLaDAMoEModel(DiffusionModelBase):
    """
    LLaDA-MoE (Mixture of Experts) model.
    Uses different mask_id (156895) compared to base LLaDA (126336).
    """
    
    # Valid parameters for LLaDA-MoE (same as base LLaDA)
    VALID_PARAMS = {
        'input_ids', 'attention_mask', 'max_new_tokens', 'num_diffusion_steps',
        'block_length', 'temperature', 'cfg_scale', 'remasking', 'mask_id',
        'logits_eos_inf', 'confidence_eos_eot_inf'
    }
    
    # Parameters that are NOT supported by LLaDA-MoE (Dream-specific)
    INVALID_PARAMS = {'alg', 'alg_temp', 'steps', 'top_p', 'top_k'}
    
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 128,
        num_diffusion_steps: int = 128,
        block_length: int = 32,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        remasking: str = 'low_confidence',
        mask_id: int = 156895,  # Different mask_id for LLaDA-MoE
        logits_eos_inf: bool = False,
        confidence_eos_eot_inf: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate using LLaDA-MoE's diffusion sampling process.
        
        LLaDA-MoE uses mask_id=156895 instead of 126336.
        
        Args:
            input_ids: Input token IDs (prompt)
            attention_mask: Attention mask for the input
            max_new_tokens: Number of tokens to generate
            num_diffusion_steps: Sampling steps, less than or equal to max_new_tokens
            block_length: Block length, less than or equal to max_new_tokens
            temperature: Categorical distribution sampling temperature
            cfg_scale: Unsupervised classifier-free guidance scale
            remasking: Remasking strategy ('low_confidence' or 'random')
            mask_id: The token id of [MASK] is 156895 for LLaDA-MoE
            logits_eos_inf: Whether to set the logits of EOS token to -inf
            confidence_eos_eot_inf: Whether to set the confidence of EOS and EoT token to -inf
            **kwargs: Additional arguments (will be validated)
        """
        # Validate and warn about invalid parameters
        if kwargs:
            invalid_params = set(kwargs.keys()) & self.INVALID_PARAMS
            if invalid_params:
                print(f"WARNING: LLaDA-MoE does not support these parameters (ignoring): {invalid_params}")
            
            # Warn about any unrecognized parameters
            unrecognized = set(kwargs.keys()) - self.VALID_PARAMS - self.INVALID_PARAMS
            if unrecognized:
                print(f"WARNING: Unrecognized parameters for LLaDA-MoE (ignoring): {unrecognized}")
        
        prompt = input_ids
        gen_length = max_new_tokens
        steps = num_diffusion_steps
        
        # Initialize full sequence with mask tokens
        x = torch.full(
            (prompt.shape[0], prompt.shape[1] + gen_length),
            mask_id,
            dtype=torch.long
        ).to(self.model.device)
        x[:, :prompt.shape[1]] = prompt.clone()

        # Extend attention mask if provided
        if attention_mask is not None:
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((prompt.shape[0], gen_length), dtype=attention_mask.dtype, device=self.model.device)
            ], dim=-1)

        prompt_index = (x != mask_id)

        assert gen_length % block_length == 0
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0
        steps_per_block = steps // num_blocks

        for num_block in range(num_blocks):
            block_mask_index = (
                x[:, prompt.shape[1] + num_block * block_length: prompt.shape[1] + (num_block + 1) * block_length]
                == mask_id
            )
            num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)
            
            for i in range(steps_per_block):
                mask_index = (x == mask_id)
                
                # Classifier-free guidance
                if cfg_scale > 0.:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    
                    if attention_mask is not None:
                        attention_mask_ = torch.cat([attention_mask, attention_mask], dim=0)
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

                if logits_eos_inf:
                    logits[:, :, 126081] = -torch.inf

                logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l
                
                if confidence_eos_eot_inf:
                    logits_with_noise[:, :, 126081] = logits[:, :, 126348] = -torch.inf

                if remasking == 'low_confidence':
                    p = F.softmax(logits, dim=-1)
                    x0_p = torch.squeeze(
                        torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
                    )  # b, l
                elif remasking == 'random':
                    x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
                else:
                    raise NotImplementedError(remasking)

                x0_p[:, prompt.shape[1] + (num_block + 1) * block_length:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
                for j in range(confidence.shape[0]):
                    _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j, i])
                    transfer_index[j, select_index] = True
                x[transfer_index] = x0[transfer_index]

        return x


class DreamModel(DiffusionModelBase):
    """Dream diffusion model."""
    
    # Valid parameters for Dream
    VALID_PARAMS = {
        'input_ids', 'inputs', 'attention_mask', 'max_new_tokens', 
        'num_diffusion_steps', 'temperature', 'alg', 'alg_temp', 
        'steps', 'top_p', 'top_k'
    }
    
    # Parameters that are NOT supported by Dream (LLaDA-specific)
    INVALID_PARAMS = {
        'cfg_scale', 'remasking', 'block_length', 'mask_id',
        'logits_eos_inf', 'confidence_eos_eot_inf'
    }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 0.2,  # Official default is 0.2
        num_diffusion_steps: int = 64,
        attention_mask: Optional[torch.Tensor] = None,
        steps: Optional[int] = None,  # Dream uses 'steps' not 'num_diffusion_steps'
        **kwargs
    ) -> torch.Tensor:
        """
        Generate using Dream's diffusion_generate method.
        
        Dream models use 'inputs' parameter instead of 'input_ids'.
        
        Args:
            input_ids: Input token IDs (will be passed as 'inputs' to Dream)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature
            num_diffusion_steps: Number of diffusion steps (converted to 'steps' for Dream)
            attention_mask: Attention mask
            steps: Dream-specific parameter (overrides num_diffusion_steps if provided)
            **kwargs: Additional Dream-specific parameters (alg, alg_temp, top_p, top_k)
        """
        if not hasattr(self.model, 'diffusion_generate'):
            raise AttributeError(
                f"Model {self.model_path} does not have 'diffusion_generate' method. "
                "Make sure you're using a Dream model."
            )
        
        # Validate and warn about invalid parameters
        if kwargs:
            invalid_params = set(kwargs.keys()) & self.INVALID_PARAMS
            if invalid_params:
                print(f"WARNING: Dream does not support these parameters (ignoring): {invalid_params}")
                # Filter out invalid parameters
                for param in invalid_params:
                    kwargs.pop(param)
            
            # Warn about any unrecognized parameters
            unrecognized = set(kwargs.keys()) - self.VALID_PARAMS - self.INVALID_PARAMS
            if unrecognized:
                print(f"WARNING: Unrecognized parameters for Dream (ignoring): {unrecognized}")
        
        # Dream's diffusion_generate signature (from official repo):
        # diffusion_generate(input_ids, attention_mask=None, max_new_tokens=..., steps=..., ...)
        # Build kwargs dict for named parameters only
        gen_kwargs = {
            'max_new_tokens': max_new_tokens,
            'steps': steps if steps is not None else num_diffusion_steps,
            'temperature': temperature,
        }
        
        if attention_mask is not None:
            gen_kwargs['attention_mask'] = attention_mask
        
        # Add remaining Dream-specific parameters (alg, alg_temp, top_p, top_k)
        gen_kwargs.update(kwargs)
        
        # Pass input_ids as first positional argument (following official example)
        return self.model.diffusion_generate(input_ids, **gen_kwargs)


def create_model(model, tokenizer, model_path: str) -> DiffusionModelBase:
    """
    Factory function to create the appropriate model class based on model_path.
    
    Args:
        model: The loaded HuggingFace model
        tokenizer: The tokenizer
        model_path: Path or identifier of the model
        
    Returns:
        An instance of the appropriate model class
    """
    model_path_lower = model_path.lower()
    
    # Dream models
    if 'dream' in model_path_lower or hasattr(model, 'diffusion_generate'):
        print(f"Creating DreamModel for {model_path}")
        return DreamModel(model, tokenizer, model_path)
    
    # LLaDA-MoE models (check before base LLaDA)
    if 'llada-moe' in model_path_lower:
        print(f"Creating LLaDAMoEModel for {model_path}")
        return LLaDAMoEModel(model, tokenizer, model_path)
    
    # Base LLaDA models
    if 'llada' in model_path_lower:
        print(f"Creating LLaDAModel for {model_path}")
        return LLaDAModel(model, tokenizer, model_path)
    
    raise ValueError(f"Model {model_path} not supported")
