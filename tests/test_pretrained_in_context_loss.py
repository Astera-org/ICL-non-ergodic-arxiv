import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F

# Add project root to sys.path to allow importing RandomWindowDataset
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from random_window_dataset import RandomWindowDataset, DEFAULT_PREPROCESSED_DIR, EFFECTIVE_WINDOW_SIZE # noqa
from train import ALL_CATEGORIES, select_categories # noqa

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

DEFAULT_OUTPUT_DIR = Path("./tests/outputs")

def calculate_in_context_losses(model, tokenizer, dataloader, device, num_samples_to_eval):
    """
    Calculates per-token loss for samples, varying context length.
    Returns a list of lists, where each inner list contains losses for one sample
    at increasing context lengths.
    """
    model.eval()
    all_samples_losses = []
    evaluated_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating in-context loss"):
            if evaluated_count >= num_samples_to_eval:
                break

            input_ids_batch = batch.to(device) # Batch of windows: (batch_size, EFFECTIVE_WINDOW_SIZE)

            # Perform a single forward pass for the entire batch
            outputs = model(input_ids=input_ids_batch) 
            # logits shape: (batch_size, EFFECTIVE_WINDOW_SIZE, vocab_size)
            logits_batch = outputs.logits

            for i in range(input_ids_batch.size(0)): # Iterate over samples in batch
                if evaluated_count >= num_samples_to_eval:
                    break
                
                # Logits for the current sample: (EFFECTIVE_WINDOW_SIZE, vocab_size)
                sample_logits = logits_batch[i]
                # Target tokens for the current sample: (EFFECTIVE_WINDOW_SIZE)
                sample_input_ids = input_ids_batch[i]
                
                sample_losses = []
                # We want to calculate loss for predicting token t_j using context t_0...t_{j-1}
                # The logits at sample_logits[j-1] are for predicting token sample_input_ids[j]
                # So, loop from the first prediction (predicting token 1) up to the last token.
                for j in range(1, EFFECTIVE_WINDOW_SIZE):
                    # Logits for predicting token at position j (target_token_id)
                    # These logits were generated using context up to position j-1.
                    # So, we take the logits from the (j-1)th position in the sequence.
                    pred_logits_for_token_j = sample_logits[j-1, :] # Shape: (vocab_size)
                    target_token_id = sample_input_ids[j]         # Shape: scalar
                    
                    loss = F.cross_entropy(pred_logits_for_token_j.unsqueeze(0), target_token_id.unsqueeze(0))
                    sample_losses.append(loss.item())
                
                all_samples_losses.append(sample_losses)
                evaluated_count += 1
                if evaluated_count % (max(1, num_samples_to_eval // 20)) == 0: # Log more frequently
                    logging.info(f"Evaluated {evaluated_count}/{num_samples_to_eval} samples...")
    
    if evaluated_count < num_samples_to_eval:
        logging.warning(f"Only evaluated {evaluated_count} samples, requested {num_samples_to_eval}. Dataset might be too small.")

    return all_samples_losses

def main():
    parser = argparse.ArgumentParser(description="Evaluate in-context loss of a pre-trained model on ArXiv validation data.")
    parser.add_argument("--model_name_or_path", type=str, default="EleutherAI/pythia-70m-deduped", help="Hugging Face model name or path.")
    parser.add_argument("--preprocessed_data_dir", type=Path, default=DEFAULT_PREPROCESSED_DIR, help="Directory with preprocessed ArXiv data.")
    parser.add_argument("--k_categories", type=int, default=5, help="Number of categories to sample from for the validation set (uses seed 0). Set to 0 to use all categories.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of validation samples to evaluate.")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to save plots and loss data.")
    parser.add_argument("--force_cpu", action="store_true", help="Force using CPU even if CUDA is available.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for dataloader (for faster processing of samples).")


    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"icl_{args.model_name_or_path.replace('/', '_')}_k{args.k_categories}_n{args.num_samples}"

    logging.info(f"Starting in-context loss evaluation for {args.model_name_or_path}")
    logging.info(f"Parameters: {vars(args)}")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu")
    logging.info(f"Using device: {device}")

    logging.info(f"Loading model and tokenizer: {args.model_name_or_path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        model.to(device)
    except Exception as e:
        logging.error(f"Error loading model/tokenizer: {e}")
        return
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logging.info(f"Tokenizer pad_token set to eos_token: {tokenizer.eos_token}")


    if args.k_categories > 0:
        val_categories = select_categories(ALL_CATEGORIES, k=args.k_categories, seed=0) # Use fixed seed for val categories
        logging.info(f"Using {args.k_categories} categories for validation (seed 0): {val_categories}")
    else:
        val_categories = None # Use all categories
        logging.info("Using all available categories for validation.")

    try:
        val_dataset = RandomWindowDataset(
            preprocessed_dir=args.preprocessed_data_dir,
            split="validation",
            target_categories=val_categories
        )
    except FileNotFoundError:
        logging.error(f"Preprocessed data not found in {args.preprocessed_data_dir}. Please run fetch_arxiv.py first.")
        return
    except ValueError as e:
        logging.error(f"Error initializing dataset: {e}")
        return
        
    if len(val_dataset) == 0:
        logging.error("Validation dataset is empty. Cannot proceed.")
        return

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False) # Shuffle false for val

    logging.info(f"Calculating in-context losses for {args.num_samples} samples...")
    all_samples_losses = calculate_in_context_losses(model, tokenizer, val_dataloader, device, args.num_samples)
    
    if not all_samples_losses:
        logging.error("No losses were calculated. Exiting.")
        return

    # Transpose to get losses per position: losses_at_pos[j] = list of losses for all samples at context length j+1
    losses_at_pos = np.array(all_samples_losses).T 
    avg_loss_per_pos = np.mean(losses_at_pos, axis=1)
    std_loss_per_pos = np.std(losses_at_pos, axis=1)

    # Context length for x-axis (token position being predicted, starting from 2nd token)
    context_positions = np.arange(1, EFFECTIVE_WINDOW_SIZE) # Goes from 1 to EFFECTIVE_WINDOW_SIZE-1

    # Calculate overall mean loss across all positions for the evaluated samples
    overall_mean_loss = np.mean(avg_loss_per_pos)
    logging.info(f"Overall mean loss across all context positions: {overall_mean_loss:.4f}")

    # Plotting
    plt.figure(figsize=(12, 7))
    plt.plot(context_positions, avg_loss_per_pos, marker='o', linestyle='-', label="Loss at context position")
    plt.fill_between(context_positions, avg_loss_per_pos - std_loss_per_pos, avg_loss_per_pos + std_loss_per_pos, alpha=0.2, label="Std Dev")
    
    # Add horizontal line for overall mean loss
    plt.axhline(overall_mean_loss, color='r', linestyle='--', label=f"Mean Loss over all positions: {overall_mean_loss:.4f}")
    
    plt.xlabel("Context Length (Number of preceding tokens)")
    plt.ylabel("Average NLL Loss for Next Token")
    plt.title(f"In-Context Learning: Average Loss vs. Context Length\nModel: {args.model_name_or_path}")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xticks(np.arange(0, EFFECTIVE_WINDOW_SIZE, step=max(1, EFFECTIVE_WINDOW_SIZE//20))) # Adjust tick step
    plt.legend() # Add legend to show labels
    plt.tight_layout()
    
    plot_path = args.output_dir / f"{run_name}_in_context_loss.png"
    plt.savefig(plot_path)
    logging.info(f"Saved plot to {plot_path}")
    plt.close()

    # Save raw average losses and std
    loss_data = {
        "model_name_or_path": args.model_name_or_path,
        "k_categories": args.k_categories,
        "num_samples_evaluated": len(all_samples_losses),
        "context_positions": context_positions.tolist(),
        "avg_loss_per_position": avg_loss_per_pos.tolist(),
        "std_loss_per_position": std_loss_per_pos.tolist(),
        "overall_mean_loss": overall_mean_loss, # Add overall mean loss to data
        "all_samples_losses": all_samples_losses # List of lists
    }
    loss_data_path = args.output_dir / f"{run_name}_loss_data.json"
    with open(loss_data_path, 'w') as f:
        json.dump(loss_data, f, indent=4)
    logging.info(f"Saved loss data to {loss_data_path}")

    logging.info("In-context loss evaluation complete.")

if __name__ == "__main__":
    main() 