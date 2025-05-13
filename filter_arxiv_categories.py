import json
import numpy as np
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pickle
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_papers_by_category(preprocessed_dir: Path, categories: List[str] = None) -> Tuple[Dict[str, List[Dict]], Dict[str, int]]:
    """Get paper metadata filtered by categories.
    
    Args:
        preprocessed_dir: Directory containing preprocessed ArXiv data
        categories: List of categories to filter (None = all categories)
        
    Returns:
        Tuple with (papers_by_category, token_counts_by_category)
    """
    index_path = preprocessed_dir / 'index.jsonl'
    papers_by_category = {}
    token_counts = {}
    
    # Load index data
    logging.info(f"Loading index from {index_path}")
    with open(index_path, 'r') as f:
        for line in f:
            paper = json.loads(line)
            category = paper['cat']
            
            # Skip if not in requested categories
            if categories and category not in categories:
                continue
                
            # Initialize if first time seeing this category
            if category not in papers_by_category:
                papers_by_category[category] = []
                token_counts[category] = 0
                
            # Add paper to category and update token count
            papers_by_category[category].append(paper)
            token_counts[category] += paper['length']
    
    # Print statistics
    for cat, papers in sorted(papers_by_category.items(), key=lambda x: token_counts[x[0]], reverse=True):
        logging.info(f"Category {cat}: {len(papers)} papers, {token_counts[cat]:,} tokens")
        
    return papers_by_category, token_counts

def create_category_specific_dataset(
    preprocessed_dir: Path,
    output_dir: Path,
    categories: List[str],
    include_splits: bool = True
) -> Dict[str, Any]:
    """Create a filtered version of the dataset for specific categories.
    
    Args:
        preprocessed_dir: Directory with original preprocessed data
        output_dir: Directory to save filtered data
        categories: List of categories to include
        include_splits: Whether to maintain train/val/test splits
        
    Returns:
        Dictionary with metadata about the created dataset
    """
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get papers by category
    papers_by_category, token_counts = get_papers_by_category(preprocessed_dir, categories)
    
    if not papers_by_category:
        logging.error(f"No papers found for categories: {categories}")
        return None
    
    # Flatten papers list
    all_papers = []
    for cat_papers in papers_by_category.values():
        all_papers.extend(cat_papers)
        
    # Load original tokens
    tokens_path = preprocessed_dir / 'tokens.bin'
    logging.info(f"Memory-mapping original tokens from {tokens_path}")
    all_tokens = np.memmap(tokens_path, dtype=np.uint16, mode='r')
    
    # Load original splits if needed
    if include_splits:
        splits_path = preprocessed_dir / 'splits.json'
        logging.info(f"Loading splits from {splits_path}")
        with open(splits_path, 'r') as f:
            original_splits = json.load(f)
    
    # Create new tokens array, index, and splits
    logging.info("Creating new filtered dataset")
    new_tokens = []
    new_index = []
    new_splits = {"train": [], "validation": [], "test": []} if include_splits else None
    
    current_offset = 0
    for paper in all_papers:
        paper_id = paper['paper_id']
        orig_offset = paper['offset']
        length = paper['length']
        
        # Extract tokens for this paper
        paper_tokens = all_tokens[orig_offset:orig_offset + length]
        
        # Add to new tokens list
        new_tokens.extend(paper_tokens)
        
        # Create new index entry
        new_index.append({
            "paper_id": paper_id,
            "cat": paper['cat'],
            "offset": current_offset,
            "length": length
        })
        
        # Add to splits if keeping them
        if include_splits:
            for split_name, paper_ids in original_splits.items():
                if paper_id in paper_ids:
                    new_splits[split_name].append(paper_id)
                    break
        
        # Update offset for next paper
        current_offset += length
    
    # Save new tokens
    new_tokens_path = output_dir / 'tokens.bin'
    logging.info(f"Saving {len(new_tokens):,} tokens to {new_tokens_path}")
    np.array(new_tokens, dtype=np.uint16).tofile(new_tokens_path)
    
    # Save new index
    new_index_path = output_dir / 'index.jsonl'
    logging.info(f"Saving index with {len(new_index)} papers to {new_index_path}")
    with open(new_index_path, 'w') as f:
        for paper in new_index:
            f.write(json.dumps(paper) + '\n')
    
    # Save new splits if needed
    if include_splits:
        new_splits_path = output_dir / 'splits.json'
        logging.info(f"Saving splits to {new_splits_path}")
        with open(new_splits_path, 'w') as f:
            json.dump(new_splits, f, indent=2)
    
    # Save metadata
    metadata = {
        "categories": categories,
        "paper_count": len(new_index),
        "token_count": len(new_tokens),
        "token_counts_by_category": token_counts,
        "paper_counts_by_category": {cat: len(papers) for cat, papers in papers_by_category.items()},
        "include_splits": include_splits
    }
    
    metadata_path = output_dir / 'metadata.json'
    logging.info(f"Saving metadata to {metadata_path}")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logging.info(f"Successfully created filtered dataset in {output_dir}")
    return metadata

def create_predefined_category_groups():
    """Create predefined category groupings for experiments"""
    groups = {
        "math": ["math.ST", "math.GR", "math.AC"],
        "theory_cs": ["cs.DS", "cs.IT"],
        "systems": ["cs.SY", "cs.CE"],
        "ai_compute": ["cs.AI", "cs.CV", "cs.NE", "cs.PL"]
    }
    
    # Print group token counts
    preprocessed_dir = Path("./preprocessed_arxiv")
    for group_name, cats in groups.items():
        _, token_counts = get_papers_by_category(preprocessed_dir, cats)
        total_tokens = sum(token_counts.values())
        logging.info(f"Group '{group_name}' ({', '.join(cats)}): {total_tokens:,} tokens")
    
    return groups

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter ArXiv dataset by categories for pretraining experiments")
    parser.add_argument("--preprocessed_dir", type=Path, default=Path("./preprocessed_arxiv"),
                       help="Directory with original preprocessed ArXiv data")
    parser.add_argument("--output_dir", type=Path, required=True,
                       help="Directory to save filtered output")
    parser.add_argument("--categories", type=str, nargs="+", 
                       help="Space-separated list of categories to include (e.g., 'cs.AI cs.CV')")
    parser.add_argument("--group", type=str, choices=["math", "theory_cs", "systems", "ai_compute"],
                       help="Use a predefined category group instead of listing categories")
    parser.add_argument("--all", action="store_true",
                       help="Show stats for all categories without filtering")
    parser.add_argument("--no_splits", action="store_true",
                       help="Don't maintain train/val/test splits in output")
    
    args = parser.parse_args()
    
    # Handle predefined groups
    groups = create_predefined_category_groups()
    
    if args.all:
        # Just show stats for all categories
        get_papers_by_category(args.preprocessed_dir)
    elif args.group:
        # Use a predefined group
        create_category_specific_dataset(
            args.preprocessed_dir,
            args.output_dir,
            groups[args.group],
            not args.no_splits
        )
    elif args.categories:
        # Use user-provided categories
        create_category_specific_dataset(
            args.preprocessed_dir,
            args.output_dir,
            args.categories,
            not args.no_splits
        )
    else:
        logging.error("You must specify either --categories, --group, or --all") 