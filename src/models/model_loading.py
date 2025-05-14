from omegaconf import DictConfig

from .micro_decoder_model import MicroDecoderConfig, MicroDecoderModel
from ..tokenizer_utils import load_tokenizer_from_config # Assuming tokenizer_utils is in src/

def load_micro_decoder_from_config(model_cfg: DictConfig, global_cfg: DictConfig) -> MicroDecoderModel:
    """
    Loads the MicroDecoderModel from Hydra configuration.

    Args:
        model_cfg: The Hydra configuration specific to this model 
                   (e.g., contents of configs/model/micro_decoder.yaml, 
                    accessed via cfg.model in the main script).
        global_cfg: The global Hydra configuration object, expected to contain
                    tokenizer path under global_cfg.model.custom_tokenizer_path.

    Returns:
        An instance of MicroDecoderModel.
    """
    
    # 1. Load the tokenizer to get vocab_size and pad_token_id
    # The tokenizer path should be in the global config, e.g., global_cfg.model.custom_tokenizer_path
    # or global_cfg.model.tokenizer_name for HF tokenizers.
    tokenizer = load_tokenizer_from_config(global_cfg.model) # Pass the model part of global_cfg
    
    vocab_size = tokenizer.get_vocab_size()
    pad_token_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None else None
    # If using a custom tokenizer from `tokenizers` library, pad_token_id might not be directly set
    # unless you explicitly added a pad token and configured it. 
    # For BPE, often there isn't a formal pad token unless specified post-training.
    # PretrainedConfig handles pad_token_id=None gracefully.

    # 2. Get architecture parameters from the model-specific config
    architecture_params = model_cfg.get("architecture")
    if architecture_params is None:
        raise ValueError("Model configuration (model_cfg) must contain an 'architecture' section.")

    # 3. Instantiate MicroDecoderConfig
    decoder_config = MicroDecoderConfig(
        vocab_size=vocab_size,
        pad_token_id=pad_token_id,
        max_position_embeddings=architecture_params.max_position_embeddings,
        d_model=architecture_params.d_model,
        n_layers=architecture_params.n_layers,
        n_heads=architecture_params.n_heads,
        d_ff=architecture_params.d_ff,
        hidden_act=architecture_params.hidden_act,
        hidden_dropout_prob=architecture_params.hidden_dropout_prob,
        attention_probs_dropout_prob=architecture_params.attention_probs_dropout_prob,
        tie_word_embeddings=architecture_params.tie_word_embeddings,
        layer_norm_eps=architecture_params.layer_norm_eps,
        initializer_range=architecture_params.initializer_range
    )

    # 4. Instantiate MicroDecoderModel
    model = MicroDecoderModel(decoder_config)

    # 5. Return the initialized model
    return model 