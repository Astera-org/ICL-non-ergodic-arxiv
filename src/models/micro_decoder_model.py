import math
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import PretrainedConfig, GPT2Config
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging
from transformers.models.gpt2.modeling_gpt2 import GPT2Attention, GPT2MLP

logger = logging.get_logger(__name__)

class MicroDecoderConfig(PretrainedConfig):
    """
    Configuration class for MicroDecoderModel.
    """
    model_type = "micro_decoder"
    attribute_map = {
        "hidden_size": "d_model",
        "num_hidden_layers": "n_layers",
        "num_attention_heads": "n_heads",
        "intermediate_size": "d_ff",
    }

    def __init__(
        self,
        vocab_size=32000,
        max_position_embeddings=100,
        d_model=256,
        n_layers=6,
        n_heads=4,
        d_ff=1024,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        tie_word_embeddings=True,
        layer_norm_eps=1e-5,
        initializer_range=0.02,
        pad_token_id=None,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.tie_word_embeddings = tie_word_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        super().__init__(pad_token_id=pad_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs)

class DecoderBlock(nn.Module):
    """
    A single Transformer Decoder block, using GPT2Attention and GPT2MLP.
    """
    def __init__(self, config: MicroDecoderConfig, layer_idx: int | None = None):
        super().__init__()
        # Create a temporary GPT2Config to pass to GPT2Attention and GPT2MLP
        # This adapts our MicroDecoderConfig names to GPT2-expected names.
        gpt2_module_config = GPT2Config(
            n_embd=config.d_model,
            n_head=config.n_heads,
            n_layer=config.n_layers, # Not strictly needed per block, but good for consistency
            n_positions=config.max_position_embeddings, # Max sequence length for attention
            activation_function=config.hidden_act,
            resid_pdrop=config.hidden_dropout_prob, # Dropout after FFN and MHA
            embd_pdrop=config.hidden_dropout_prob,  # Dropout for embeddings (not used here directly)
            attn_pdrop=config.attention_probs_dropout_prob, # Dropout in MHA
            layer_norm_epsilon=config.layer_norm_eps,
            initializer_range=config.initializer_range,
            # Pass d_ff as n_inner for GPT2MLP. If n_inner is not set, GPT2MLP defaults to 4*n_embd.
            n_inner=config.d_ff, 
            vocab_size=config.vocab_size # For consistency, though not directly used by block modules
        )

        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        # is_cross_attention=False because it's self-attention in a decoder
        self.attn = GPT2Attention(config=gpt2_module_config, is_cross_attention=False, layer_idx=layer_idx)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.mlp = GPT2MLP(intermediate_size=config.d_ff, config=gpt2_module_config)
        # GPT2MLP applies dropout internally after the first FC layer if configured via activation_function related dropout
        # and resid_pdrop is applied by GPT2Block after the MLP. Here, our main hidden_dropout_prob handles it.

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None):
        # hidden_states: (batch_size, seq_len, d_model)
        # attention_mask: (batch_size, 1, 1, seq_len) - from MicroDecoderModel.forward()
        # GPT2Attention expects layer_past=None, attention_mask, head_mask=None, use_cache=False, output_attentions=False
        
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        # GPT2Attention returns a tuple: (attn_output, present_key_value, *attentions)
        # We only need attn_output for a standard decoder block pass.
        attn_outputs = self.attn(
            hidden_states,
            layer_past=None, # Not using KV caching for simplicity here
            attention_mask=attention_mask,
            head_mask=None, # Not using head masking
            use_cache=False, # Not using KV caching
            output_attentions=False
        )
        attn_output = attn_outputs[0]  # output_attn is (batch_size, seq_len, d_model)
        
        # Residual connection after attention
        hidden_states = attn_output + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        # GPT2MLP expects hidden_states
        feed_forward_hidden_states = self.mlp(hidden_states)
        
        # Residual connection after MLP
        hidden_states = feed_forward_hidden_states + residual
        
        return hidden_states # (batch_size, seq_len, d_model)

class MicroDecoderModel(PreTrainedModel):
    """
    Micro Decoder model implementation.
    """
    config_class = MicroDecoderConfig

    def __init__(self, config: MicroDecoderConfig):
        super().__init__(config)
        self.config = config

        self.token_embeddings = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.d_model)
        self.emb_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.layers = nn.ModuleList([DecoderBlock(config, layer_idx=i) for i in range(config.n_layers)])
        self.final_ln = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

        # LM head is typically not defined here if tie_word_embeddings=True, 
        # as it's handled by get_output_embeddings() and _init_weights()
        # However, if we want to always have it as a distinct layer that can be tied:
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.token_embeddings

    def set_input_embeddings(self, new_embeddings: nn.Embedding):
        self.token_embeddings = new_embeddings

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
        # Tie output projection to token embedding
        if self.config.tie_word_embeddings and isinstance(module, nn.Linear) and module is self.lm_head:
            logger.info("Tying word embeddings for MicroDecoderModel")
            self.lm_head.weight = self.token_embeddings.weight

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None, # Typically (batch_size, seq_len)
        position_ids: torch.LongTensor | None = None,
        # head_mask: torch.FloatTensor | None = None, # Not implemented for simplicity
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None, # Not implemented for simplicity
        output_attentions: bool | None = None, # Not implemented for simplicity
        output_hidden_states: bool | None = None, # Not implemented for simplicity
        return_dict: bool | None = None,
    ):
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.token_embeddings(input_ids)

        if position_ids is None:
            position_ids = torch.arange(0, seq_length, dtype=torch.long, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0) # .expand(batch_size, seq_length)
        
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = inputs_embeds + position_embeds
        hidden_states = self.emb_dropout(hidden_states)

        # Prepare attention mask for MHA layers if user provides a 2D mask
        # MHA expects (batch_size, n_heads, seq_len, seq_len) or (batch_size, 1, seq_len, seq_len)
        # Causal mask is handled within MultiHeadAttention based on its registered buffer
        # Here, we just adapt the user-provided padding mask if any.
        # For GPT2Attention, the causal mask is built-in.
        # The attention_mask passed to GPT2Attention should be (batch, 1, 1, seq_length) for a padding mask.
        # Values should be 0 for tokens to attend to and -large_neg_val for masked tokens.
        
        extended_attention_mask = None
        if attention_mask is not None:
            if attention_mask.dim() == 2: # (batch_size, seq_len), 1 for attend, 0 for mask
                # We need to invert and convert to additive mask for GPT2Attention
                extended_attention_mask = attention_mask[:, None, None, :].to(dtype=hidden_states.dtype) 
                extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(extended_attention_mask.dtype).min
            elif attention_mask.dim() == 4: # Assumed to be correctly shaped for GPT2Attention already
                extended_attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            else:
                raise ValueError("Attention mask should be 2D (batch, seq_len) or 4D.")

        # Decoder layers
        # all_hidden_states = () if output_hidden_states else None
        # all_self_attns = () if output_attentions else None

        for i, block in enumerate(self.layers):
            # if output_hidden_states:
            #     all_hidden_states = all_hidden_states + (hidden_states,)
            
            hidden_states = block(hidden_states, attention_mask=extended_attention_mask)
            
            # if output_attentions and len(layer_outputs) > 1:
            #     all_self_attns = all_self_attns + (layer_outputs[1],)
            # pass # Placeholder for actual block forward call -> Now implemented with block call
        
        hidden_states = self.final_ln(hidden_states)

        # if output_hidden_states:
        #     all_hidden_states = all_hidden_states + (hidden_states,)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + all_hidden_states + all_self_attns
        #     return ((loss,) + output) if loss is not None else output

        # return CausalLMOutputWithPast(
        #     loss=loss,
        #     logits=logits,
        #     past_key_values=None, # Not implemented for simplicity
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        # )
        return (loss, logits) # Simplified output for now 