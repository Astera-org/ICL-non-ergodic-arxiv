import torch, json
from pathlib import Path
from transformers import AutoTokenizer

print("Starting batch check...")
batch_path = Path('./problematic_batch.pt')
args_path = Path('./problematic_run_args.json')

if not batch_path.exists() or not args_path.exists():
    print(f"Error: Ensure {batch_path} ({batch_path.exists()}) and {args_path} ({args_path.exists()}) exist in the current directory.")
    exit(1)

try:
    batch = torch.load(batch_path)
    print(f"Loaded batch from {batch_path}")
    with open(args_path) as f:
        args = json.load(f)
    print(f"Loaded args from {args_path}")

    tok = AutoTokenizer.from_pretrained(args["model_name_or_path"])
    print(f"Loaded tokenizer {args['model_name_or_path']}")

    vocab_size = tok.vocab_size
    print(f'Vocab size: {vocab_size}')
    print(f'Batch shape: {batch.shape}, Batch dtype: {batch.dtype}')
    
    # Check range BEFORE checking min/max if possible
    batch_min = batch.min().item()
    batch_max = batch.max().item()
    print(f'Batch Min ID: {batch_min}, Batch Max ID: {batch_max}')

    bad_hi_mask = (batch >= vocab_size)
    bad_lo_mask = (batch < 0)
    
    bad_hi_indices = bad_hi_mask.nonzero(as_tuple=False)
    bad_lo_indices = bad_lo_mask.nonzero(as_tuple=False)
    
    num_bad_hi = bad_hi_indices.shape[0]
    num_bad_lo = bad_lo_indices.shape[0]
    
    print(f'Indices >= vocab_size ({vocab_size}): {num_bad_hi}')
    print(f'Indices < 0: {num_bad_lo}')

    if num_bad_hi > 0:
        print(f'  Example high indices (first 5): {bad_hi_indices[:5].tolist()}')
        # Find the actual values at these locations
        example_values = []
        for idx_pair in bad_hi_indices[:5]:
             example_values.append(batch[idx_pair[0], idx_pair[1]].item())
        print(f'  Example high ID values: {example_values}')

    if num_bad_lo > 0:
        print(f'  Example low indices (first 5): {bad_lo_indices[:5].tolist()}')
        example_values = []
        for idx_pair in bad_lo_indices[:5]:
             example_values.append(batch[idx_pair[0], idx_pair[1]].item())
        print(f'  Example low ID values: {example_values}')

    print("Batch check finished.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc() 