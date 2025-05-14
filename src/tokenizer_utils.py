"""Utilities for loading and using tokenizers, and for processing text data into tokens."""

from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from omegaconf import DictConfig
import hydra
from tokenizers import Tokenizer
from pathlib import Path

from src.logging_config import get_logger # Assuming logging_config.py is in the same directory

log = get_logger(__name__)

# Define project root if not already defined (assuming script is in src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_tokenizer_from_config(cfg: DictConfig) -> Optional[PreTrainedTokenizerFast]:
    """
    Loads a tokenizer. 
    If cfg.tokenizer_name is set, uses AutoTokenizer from Hugging Face.
    If cfg.custom_tokenizer_path is set, loads a custom tokenizer using load_custom_bpe_tokenizer.

    Args:
        cfg (DictConfig): Hydra configuration object, expected to directly contain
                          tokenizer_name or custom_tokenizer_path.

    Returns:
        An instance of PreTrainedTokenizerFast or a custom Tokenizer, or None if loading fails.
    """
    # Access directly from cfg, as cfg is assumed to be the relevant model-specific config part
    if cfg.get('tokenizer_name') and cfg.tokenizer_name:
        tokenizer_name = cfg.tokenizer_name
        log.info(f"Loading Hugging Face tokenizer: {tokenizer_name}...")
        try:
            # use_fast=True is generally recommended for HuggingFace tokenizers
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
            log.info(f"Tokenizer '{tokenizer_name}' loaded successfully.")
            
            # Set padding token if it's not already set (e.g. for GPT-2 like models)
            if tokenizer.pad_token is None:
                if tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                    log.info(f"Set pad_token to eos_token: {tokenizer.eos_token}")
                else:
                    log.warning(f"Cannot set pad_token for {tokenizer_name} as eos_token is also None.")
            
            return tokenizer
        except Exception as e:
            log.error(f"Error loading Hugging Face tokenizer {tokenizer_name}: {e}", exc_info=True)
            return None # Or re-raise depending on desired error handling
    elif cfg.get('custom_tokenizer_path'):
        custom_path_str = cfg.custom_tokenizer_path
        # Assume custom_tokenizer_path is relative to project root if not absolute
        custom_path = Path(custom_path_str) if Path(custom_path_str).is_absolute() else PROJECT_ROOT / custom_path_str
        log.info(f"Attempting to load custom tokenizer from: {custom_path}")
        return load_custom_bpe_tokenizer(tokenizer_path=custom_path)
    else:
        log.error("Tokenizer configuration missing: cfg.tokenizer_name or cfg.custom_tokenizer_path must be set.")
        return None

def load_custom_bpe_tokenizer(tokenizer_path: Path = PROJECT_ROOT / "models" / "custom_tokenizer" / "custom_bpe_tokenizer.json"):
    """Loads our custom trained BPE tokenizer from the specified JSON file."""
    if not tokenizer_path.exists():
        log.error(f"Custom tokenizer file not found at {tokenizer_path}")
        raise FileNotFoundError(f"Custom tokenizer file not found at {tokenizer_path}")
    try:
        log.info(f"Loading custom BPE tokenizer from: {tokenizer_path}")
        custom_tokenizer = Tokenizer.from_file(str(tokenizer_path))
        
        # For consistency with Hugging Face tokenizers, we can try to set a pad_token_id
        # if [PAD] is in the vocabulary. This helps in batching later.
        if "[PAD]" in custom_tokenizer.get_vocab():
            pad_id = custom_tokenizer.token_to_id("[PAD]")
            # Manually setting attributes like this on a raw Tokenizer object might not be standard.
            # It's often better to wrap it in a PreTrainedTokenizerFast or handle padding explicitly.
            # For now, let's add it if it helps downstream tasks, but be mindful.
            setattr(custom_tokenizer, 'pad_token_id', pad_id)
            log.info(f"Set custom_tokenizer.pad_token_id to: {pad_id} (for [PAD])")
        else:
            log.warning("Warning: [PAD] token not found in custom tokenizer vocab. pad_token_id not set.")
            setattr(custom_tokenizer, 'pad_token_id', None)
        
        return custom_tokenizer
    except Exception as e:
        log.error(f"Error loading custom BPE tokenizer from {tokenizer_path}: {e}", exc_info=True)
        raise

def chunk_token_ids(token_ids: List[int], chunk_size: int) -> List[List[int]]:
    """
    Chunks a list of token IDs into fixed-size chunks.
    Discards the last chunk if it's smaller than chunk_size.

    Args:
        token_ids: A list of integer token IDs.
        chunk_size: The desired size for each chunk.

    Returns:
        A list of lists, where each inner list is a chunk of token IDs
        of length chunk_size.
    """
    if not token_ids or chunk_size <= 0:
        return []

    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i : i + chunk_size]
        if len(chunk) == chunk_size: # Only add chunks that are exactly chunk_size long
            chunks.append(chunk)
    return chunks

def tokenize_document_texts(
    texts: List[str], 
    tokenizer: PreTrainedTokenizerFast,
    padding: Union[bool, str] = False,       # No padding by default at this stage
    truncation: Union[bool, str] = False,    # No truncation by default at this stage
    max_length: Optional[int] = None,        # Only relevant if truncation is True
    return_tensors: Optional[str] = None     # Typically 'pt' for PyTorch, None for lists
) -> Optional[Dict[str, List]]:
    """
    Tokenizes a list of document texts.

    Args:
        texts: A list of strings, where each string is a document.
        tokenizer: An initialized HuggingFace PreTrainedTokenizerFast.
        padding: Strategy for padding. See HuggingFace documentation.
                 Defaults to False (no padding).
        truncation: Strategy for truncation. See HuggingFace documentation.
                    Defaults to False (no truncation).
        max_length: Maximum length for truncation/padding. Only used if truncation/padding is enabled.
        return_tensors: If set, will return tensors of a particular framework ('pt', 'tf', 'np').
                        Defaults to None, returning lists of integers.

    Returns:
        A dictionary containing tokenized outputs (e.g., 'input_ids', 'attention_mask'),
        or None if an error occurs.
    """
    if not texts:
        log.warning("Received an empty list of texts for tokenization.")
        return {'input_ids': [], 'attention_mask': []} # Consistent with tokenizer output
    
    if not tokenizer:
        log.error("Tokenizer not provided for tokenization.")
        return None

    try:
        log.info(f"Tokenizing {len(texts)} documents... Padding: {padding}, Truncation: {truncation}, Max_length: {max_length}, Return_tensors: {return_tensors}")
        
        tokenized_output = tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            verbose=False # Reduce verbosity for batch processing
        )
        log.info(f"Successfully tokenized {len(texts)} documents.")
        return tokenized_output
        
    except Exception as e:
        log.error(f"Error during tokenization: {e}", exc_info=True)
        return None

@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main_test(cfg: DictConfig) -> None:
    """
    Main function for testing tokenizer utilities.
    Loads tokenizer from config and tokenizes sample texts.
    """
    from .logging_config import setup_logging # Local import for test
    setup_logging(cfg.log_level)
    log.info("--- Testing Tokenizer Utilities ---")

    tokenizer = load_tokenizer_from_config(cfg)

    if not tokenizer:
        log.error("Tokenizer loading failed. Aborting test.")
        return

    sample_texts = [
        "This is the first sample document for testing.",
        "Hydra makes configuration management easy!",
        "EleutherAI's Pythia models are powerful.",
        """A slightly longer document to see how tokenization handles it.
        It includes multiple lines and some punctuation like commas, periods, and exclamation marks!
        Let's ensure everything works as expected with the chosen tokenizer."?"""
    ]

    log.info("\n--- Test Case 1: Default (no padding, no truncation, returns lists) ---")
    tokenized_default = tokenize_document_texts(sample_texts, tokenizer)
    if tokenized_default:
        log.info(f"Number of tokenized samples: {len(tokenized_default['input_ids'])}")
        for i, ids in enumerate(tokenized_default['input_ids']):
            log.info(f"Sample {i+1} ({len(ids)} tokens): {ids[:10]}... {ids[-10:] if len(ids) > 20 else ''}")
            log.info(f"Decoded {i+1}: {tokenizer.decode(ids)}")

    log.info("\n--- Test Case 2: With padding to max_length of longest, no truncation, returns lists ---")
    tokenized_padded = tokenize_document_texts(sample_texts, tokenizer, padding='longest')
    if tokenized_padded:
        log.info(f"Number of tokenized samples: {len(tokenized_padded['input_ids'])}")
        for i, ids in enumerate(tokenized_padded['input_ids']):
            log.info(f"Sample {i+1} ({len(ids)} tokens): {ids}")
            # log.info(f"Attention Mask {i+1}: {tokenized_padded['attention_mask'][i]}")

    log.info("\n--- Test Case 3: With truncation and padding to max_length 10, returns lists ---")
    tokenized_trunc_padded = tokenize_document_texts(sample_texts, tokenizer, padding='max_length', truncation=True, max_length=10)
    if tokenized_trunc_padded:
        log.info(f"Number of tokenized samples: {len(tokenized_trunc_padded['input_ids'])}")
        for i, ids in enumerate(tokenized_trunc_padded['input_ids']):
            log.info(f"Sample {i+1} ({len(ids)} tokens): {ids}")
            # log.info(f"Attention Mask {i+1}: {tokenized_trunc_padded['attention_mask'][i]}")

    log.info("\n--- Test Case 4: Return PyTorch tensors --- ")
    tokenized_pt = tokenize_document_texts(sample_texts, tokenizer, padding='longest', return_tensors='pt')
    if tokenized_pt:
        log.info(f"input_ids shape: {tokenized_pt['input_ids'].shape}")
        log.info(f"attention_mask shape: {tokenized_pt['attention_mask'].shape}")
        log.info(f"First input_ids tensor: {tokenized_pt['input_ids'][0]}")

    log.info("\n--- Test Case 5: Empty list input ---")
    tokenized_empty = tokenize_document_texts([], tokenizer)
    if tokenized_empty:
        log.info(f"Tokenized empty list: {tokenized_empty}")

    log.info("\n--- Test Case 6: Tokenizer not provided (simulating error) ---")
    tokenize_document_texts(sample_texts, None) # type: ignore

    log.info("\n--- Test Case 7: Chunking token IDs ---")
    sample_ids_1 = list(range(250)) # Exactly 2.5 chunks of 100
    chunks_1 = chunk_token_ids(sample_ids_1, 100)
    log.info(f"Original {len(sample_ids_1)} IDs, chunk_size 100. Got {len(chunks_1)} chunks.")
    for i, chunk in enumerate(chunks_1):
        log.info(f"  Chunk {i+1} ({len(chunk)} IDs): {chunk[:5]}...{chunk[-5:]}")
    assert len(chunks_1) == 2
    assert all(len(c) == 100 for c in chunks_1)

    sample_ids_2 = list(range(199)) # Less than 2 chunks
    chunks_2 = chunk_token_ids(sample_ids_2, 100)
    log.info(f"Original {len(sample_ids_2)} IDs, chunk_size 100. Got {len(chunks_2)} chunks.")
    for i, chunk in enumerate(chunks_2):
        log.info(f"  Chunk {i+1} ({len(chunk)} IDs): {chunk[:5]}...{chunk[-5:]}")
    assert len(chunks_2) == 1
    assert all(len(c) == 100 for c in chunks_2)


    sample_ids_3 = list(range(50)) # Less than 1 chunk
    chunks_3 = chunk_token_ids(sample_ids_3, 100)
    log.info(f"Original {len(sample_ids_3)} IDs, chunk_size 100. Got {len(chunks_3)} chunks.")
    assert len(chunks_3) == 0

    sample_ids_4 = [] # Empty input
    chunks_4 = chunk_token_ids(sample_ids_4, 100)
    log.info(f"Original {len(sample_ids_4)} IDs, chunk_size 100. Got {len(chunks_4)} chunks.")
    assert len(chunks_4) == 0
    
    sample_ids_5 = list(range(200)) # Exactly 2 chunks
    chunks_5 = chunk_token_ids(sample_ids_5, 100)
    log.info(f"Original {len(sample_ids_5)} IDs, chunk_size 100. Got {len(chunks_5)} chunks.")
    assert len(chunks_5) == 2
    assert all(len(c) == 100 for c in chunks_5)

    log.info("\n--- End of Tokenizer Utilities Test ---")

if __name__ == "__main__":
    # This allows running the script directly for testing:
    # python -m src.tokenizer_utils
    main_test() 