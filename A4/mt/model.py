import os
from pathlib import Path

from transformers import GPT2LMHeadModel
import torch
import torch.nn as nn

class GPTPromptTuning:
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        tokeniser,
        prompt_str: str,
        **kwargs,
    ):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # freeze GPT params
        for param in model.parameters():
            param.requires_grad = False
            
        print("Initializing soft prompt from vocab...")
        model.initialize_soft_prompt(
            tokeniser,
            prompt_str,
        )

        return model

    def initialize_soft_prompt(
        self,
        tokeniser,
        prompt_str: str
    ):
        prompt_tokens = tokeniser.encode(prompt_str, add_special_tokens=False)
        self.n_tokens = len(prompt_tokens)
        
        self.soft_prompt = nn.Embedding(self.n_tokens, self.config.n_embd)
        
        init_prompt_value = self.transformer.wte.weight[prompt_tokens].clone().detach()
        self.soft_prompt.weight = nn.parameter.Parameter(init_prompt_value)

    def _cat_learned_embedding_to_input(self, input_ids):
        inputs_embeds = self.transformer.wte(input_ids)

        if len(list(inputs_embeds.shape)) == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        # [batch_size, n_tokens, n_embd]
        learned_embeds = self.soft_prompt.weight.repeat(inputs_embeds.size(0), 1, 1)

        inputs_embeds = torch.cat([learned_embeds, inputs_embeds], dim=1)

        return inputs_embeds

    def _extend_labels(self, labels, ignore_index=-100):
        if len(list(labels.shape)) == 1:
            labels = labels.unsqueeze(0)

        n_batches = labels.shape[0]
        return torch.cat(
            [
                torch.full((n_batches, self.n_tokens), ignore_index).to(self.device),
                labels,
            ],
            dim=1,
        )

    def _extend_attention_mask(self, attention_mask):
        if len(list(attention_mask.shape)) == 1:
            attention_mask = attention_mask.unsqueeze(0)

        n_batches = attention_mask.shape[0]
        return torch.cat(
            [torch.full((n_batches, self.n_tokens), 1).to(self.device), attention_mask],
            dim=1,
        )

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        return_dict=None,
    ):
        if input_ids is not None:
            inputs_embeds = self._cat_learned_embedding_to_input(input_ids).to(
                self.device
            )

        if labels is not None:
            labels = self._extend_labels(labels).to(self.device)

        if attention_mask is not None:
            attention_mask = self._extend_attention_mask(attention_mask).to(self.device)

        return super().forward(
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            return_dict=return_dict,
        )


class GPT2PromptTuningLM(GPTPromptTuning, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)