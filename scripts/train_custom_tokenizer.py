import os
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tqdm import tqdm

# Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models" / "custom_tokenizer"
CORPUS_FILE = DATA_DIR / "tokenizer_training_corpus.txt" # From 16.1

# Ensure model directory exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VOCAB_SIZE = 32000
SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

def configure_tokenizer_and_trainer():
    """
    Configures and returns a BPE tokenizer and its trainer.
    This fulfills Subtask 16.2.
    """
    # 1. Initialize a BPE Tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

    # 2. Configure pre-tokenization
    # Using whitespace splitting is common, and then BPE learns merges from there.
    # Metaspace is often used to handle spaces more naturally, especially for sentencepiece-like behavior.
    # tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="_", add_prefix_space=True)
    # Alternatively, for simpler whitespace splitting:
    # tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)

    # 3. Configure the decoder
    # Metaspace decoder is used if Metaspace pre_tokenizer is used.
    # tokenizer.decoder = decoders.Metaspace(replacement="_", add_prefix_space=True)
    # Alternatively, for simpler whitespace based decoding:
    # tokenizer.decoder = decoders.WordPiece(prefix="##") # or decoders.BPEDecoder()
    tokenizer.decoder = decoders.ByteLevel()

    print("Tokenizer configured with:")
    print(f"  Model: BPE (unk_token='{tokenizer.model.unk_token}')")
    print(f"  Pre-tokenizer: {tokenizer.pre_tokenizer}")
    print(f"  Decoder: {tokenizer.decoder}")

    # 4. Set up the BPE Trainer
    # min_frequency: The minimum frequency a pair should have to be merged.
    # show_progress: Displays a progress bar.
    # vocab_size: The target vocabulary size.
    # special_tokens: A list of special tokens to add to the vocabulary.
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=2, # A common default, can be tuned
        show_progress=True,
        special_tokens=SPECIAL_TOKENS
    )
    print("\nTrainer configured with:")
    print(f"  Vocab size: {trainer.vocab_size}")
    print(f"  Min frequency: {trainer.min_frequency}")
    print(f"  Special tokens: {trainer.special_tokens}")
    
    return tokenizer, trainer

def main():
    print("--- Subtask 16.2: Implement BPE Tokenizer Training Configuration ---")
    tokenizer, trainer = configure_tokenizer_and_trainer()
    print("\nConfiguration complete.")

    print("\n--- Subtask 16.3: Implement Tokenizer Training Process ---")
    if not CORPUS_FILE.exists():
        print(f"ERROR: Corpus file not found at {CORPUS_FILE}")
        print("Please run the data preparation script (e.g., scripts/prepare_tokenizer_data.py) first.")
        return

    files = [str(CORPUS_FILE)]
    print(f"Starting tokenizer training on: {files}")
    print(f"This might take a while for a corpus of size {CORPUS_FILE.stat().st_size / (1024*1024):.2f} MB...")
    
    try:
        tokenizer.train(files, trainer=trainer)
        print("Tokenizer training successfully completed.")
    except Exception as e:
        print(f"An error occurred during tokenizer training: {e}")
        return

    print("\n--- Subtask 16.4: Save and Validate the Trained Tokenizer ---")
    # The Hugging Face Tokenizer's BPE model typically saves to a single JSON file
    # that includes vocab, merges, and config.
    tokenizer_save_path = MODEL_DIR / "custom_bpe_tokenizer.json"
    
    try:
        tokenizer.save(str(tokenizer_save_path))
        print(f"Tokenizer model saved to {tokenizer_save_path}")
    except Exception as e:
        print(f"An error occurred while saving the tokenizer: {e}")
        return

    # Test loading and encoding/decoding (simplified test here)
    try:
        print("\nValidating saved tokenizer...")
        loaded_tokenizer = Tokenizer.from_file(str(tokenizer_save_path))
        sample_text = "This is a test sentence for our new tokenizer, covering cs.DS and math.GR topics."
        
        print(f"Original sample text: {sample_text}")
        
        encoding = loaded_tokenizer.encode(sample_text)
        print(f"Encoded IDs: {encoding.ids}")
        print(f"Encoded Tokens: {encoding.tokens}")
        
        decoded_text = loaded_tokenizer.decode(encoding.ids)
        print(f"Decoded text: {decoded_text}")
        
        if sample_text == decoded_text:
            print("Validation successful: Decoded text matches original.")
        else:
            print("Validation warning: Decoded text differs from original. This can sometimes happen due to pre-tokenization specifics (e.g., prefix spaces).")
            print(f"  Original: '{sample_text}'")
            print(f"  Decoded:  '{decoded_text}'")

    except Exception as e:
        print(f"An error occurred during tokenizer validation: {e}")

if __name__ == "__main__":
    main() 