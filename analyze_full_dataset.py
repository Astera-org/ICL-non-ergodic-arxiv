import json
import numpy as np
import logging
from pathlib import Path
import argparse

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Sample data token counts from our limited analysis
FULL_DATASET_TOTAL_TOKENS = 606_491_363  # From previous analysis of the full S3 dataset

# Category distribution percentages from our sample
CATEGORY_DISTRIBUTION = {
    "cs.DS": 0.1599,
    "math.ST": 0.0778,
    "math.GR": 0.0815,
    "cs.PL": 0.1932,
    "cs.IT": 0.0778,
    "math.AC": 0.0396,
    "cs.SY": 0.0952,
    "cs.AI": 0.0934,
    "cs.NE": 0.0669,
    "cs.CE": 0.0799,
    "cs.CV": 0.0348
}

def estimate_full_dataset_distribution():
    """Estimate the distribution of the full dataset based on our sample"""
    logging.info("Estimated token counts for FULL ArXiv dataset on S3:")
    logging.info(f"Total tokens in full dataset: {FULL_DATASET_TOTAL_TOKENS:,}")
    
    # Calculate token counts for each category
    estimated_tokens = {}
    for cat, percentage in CATEGORY_DISTRIBUTION.items():
        tokens = int(FULL_DATASET_TOTAL_TOKENS * percentage)
        estimated_tokens[cat] = tokens
    
    # Print sorted by token count
    for cat, tokens in sorted(estimated_tokens.items(), key=lambda x: x[1], reverse=True):
        logging.info(f"{cat}: {tokens:,} tokens ({CATEGORY_DISTRIBUTION[cat]*100:.2f}% of dataset)")
    
    return estimated_tokens

def define_balanced_groups(estimated_tokens):
    """Define semantically meaningful groups with similar token counts"""
    
    logging.info("\n=== 3 Semantically Balanced Groups for FULL Dataset ===")
    
    # Group 1: Mathematics (math.ST, math.GR, math.AC)
    group1 = ["math.ST", "math.GR", "math.AC"]
    group1_tokens = sum(estimated_tokens[cat] for cat in group1)
    logging.info(f"Group 1: Mathematics ({', '.join(group1)})")
    logging.info(f"  {group1_tokens:,} tokens ({group1_tokens/FULL_DATASET_TOTAL_TOKENS*100:.2f}% of total)")
    
    # Group 2: Core CS (cs.DS, cs.IT, cs.SY, cs.CE)
    group2 = ["cs.DS", "cs.IT", "cs.SY", "cs.CE"]
    group2_tokens = sum(estimated_tokens[cat] for cat in group2)
    logging.info(f"Group 2: Core Computer Science ({', '.join(group2)})")
    logging.info(f"  {group2_tokens:,} tokens ({group2_tokens/FULL_DATASET_TOTAL_TOKENS*100:.2f}% of total)")
    
    # Group 3: AI & Applications (cs.AI, cs.NE, cs.CV, cs.PL)
    group3 = ["cs.AI", "cs.NE", "cs.CV", "cs.PL"]
    group3_tokens = sum(estimated_tokens[cat] for cat in group3)
    logging.info(f"Group 3: AI & Applications ({', '.join(group3)})")
    logging.info(f"  {group3_tokens:,} tokens ({group3_tokens/FULL_DATASET_TOTAL_TOKENS*100:.2f}% of total)")
    
    # Output group statistics
    groups = [group1_tokens, group2_tokens, group3_tokens]
    logging.info(f"\n3-group statistics:")
    logging.info(f"Min tokens: {min(groups):,}")
    logging.info(f"Max tokens: {max(groups):,}")
    logging.info(f"Max/Min ratio: {max(groups)/min(groups):.2f}")
    logging.info(f"Std dev: {np.std(groups):,.0f} ({np.std(groups)/np.mean(groups)*100:.2f}% of mean)")
    
    logging.info("\n=== 4 Semantically Balanced Groups for FULL Dataset ===")
    
    # Group 1: Pure Mathematics (math.GR, math.AC)
    group1 = ["math.GR", "math.AC"]
    group1_tokens = sum(estimated_tokens[cat] for cat in group1)
    logging.info(f"Group 1: Pure Mathematics ({', '.join(group1)})")
    logging.info(f"  {group1_tokens:,} tokens ({group1_tokens/FULL_DATASET_TOTAL_TOKENS*100:.2f}% of total)")
    
    # Group 2: Applied Math & Information Theory (math.ST, cs.IT)
    group2 = ["math.ST", "cs.IT"]
    group2_tokens = sum(estimated_tokens[cat] for cat in group2)
    logging.info(f"Group 2: Applied Math & Information Theory ({', '.join(group2)})")
    logging.info(f"  {group2_tokens:,} tokens ({group2_tokens/FULL_DATASET_TOTAL_TOKENS*100:.2f}% of total)")
    
    # Group 3: Algorithms & Systems (cs.DS, cs.SY, cs.CE)
    group3 = ["cs.DS", "cs.SY", "cs.CE"]
    group3_tokens = sum(estimated_tokens[cat] for cat in group3)
    logging.info(f"Group 3: Algorithms & Systems ({', '.join(group3)})")
    logging.info(f"  {group3_tokens:,} tokens ({group3_tokens/FULL_DATASET_TOTAL_TOKENS*100:.2f}% of total)")
    
    # Group 4: AI & Applications (cs.AI, cs.NE, cs.CV, cs.PL)
    group4 = ["cs.AI", "cs.NE", "cs.CV", "cs.PL"]
    group4_tokens = sum(estimated_tokens[cat] for cat in group4)
    logging.info(f"Group 4: AI & Applications ({', '.join(group4)})")
    logging.info(f"  {group4_tokens:,} tokens ({group4_tokens/FULL_DATASET_TOTAL_TOKENS*100:.2f}% of total)")
    
    # Output group statistics
    groups = [group1_tokens, group2_tokens, group3_tokens, group4_tokens]
    logging.info(f"\n4-group statistics:")
    logging.info(f"Min tokens: {min(groups):,}")
    logging.info(f"Max tokens: {max(groups):,}")
    logging.info(f"Max/Min ratio: {max(groups)/min(groups):.2f}")
    logging.info(f"Std dev: {np.std(groups):,.0f} ({np.std(groups)/np.mean(groups)*100:.2f}% of mean)")
    
    logging.info("\n=== 5 Nearly Equal-Sized Groups (Modified for Balance) ===")
    
    # For 5 groups, we need to redesign to get more even token distribution
    group1 = ["cs.PL"]  # Largest category on its own
    group1_tokens = sum(estimated_tokens[cat] for cat in group1)
    
    group2 = ["cs.DS"]  # Second largest category on its own
    group2_tokens = sum(estimated_tokens[cat] for cat in group2)
    
    group3 = ["math.GR", "math.AC"]  # Pure math
    group3_tokens = sum(estimated_tokens[cat] for cat in group3)
    
    group4 = ["cs.IT", "cs.CE"]  # Theory and Engineering
    group4_tokens = sum(estimated_tokens[cat] for cat in group4)
    
    group5 = ["cs.AI", "cs.NE", "cs.CV", "cs.SY", "math.ST"]  # Mixed category
    group5_tokens = sum(estimated_tokens[cat] for cat in group5)
    
    logging.info(f"Group 1: Programming Languages ({', '.join(group1)})")
    logging.info(f"  {group1_tokens:,} tokens ({group1_tokens/FULL_DATASET_TOTAL_TOKENS*100:.2f}% of total)")
    
    logging.info(f"Group 2: Algorithms ({', '.join(group2)})")
    logging.info(f"  {group2_tokens:,} tokens ({group2_tokens/FULL_DATASET_TOTAL_TOKENS*100:.2f}% of total)")
    
    logging.info(f"Group 3: Pure Mathematics ({', '.join(group3)})")
    logging.info(f"  {group3_tokens:,} tokens ({group3_tokens/FULL_DATASET_TOTAL_TOKENS*100:.2f}% of total)")
    
    logging.info(f"Group 4: Theory & Engineering ({', '.join(group4)})")
    logging.info(f"  {group4_tokens:,} tokens ({group4_tokens/FULL_DATASET_TOTAL_TOKENS*100:.2f}% of total)")
    
    logging.info(f"Group 5: AI, Systems & Stats ({', '.join(group5)})")
    logging.info(f"  {group5_tokens:,} tokens ({group5_tokens/FULL_DATASET_TOTAL_TOKENS*100:.2f}% of total)")
    
    # Output group statistics
    groups = [group1_tokens, group2_tokens, group3_tokens, group4_tokens, group5_tokens]
    logging.info(f"\n5-group statistics:")
    logging.info(f"Min tokens: {min(groups):,}")
    logging.info(f"Max tokens: {max(groups):,}")
    logging.info(f"Max/Min ratio: {max(groups)/min(groups):.2f}")
    logging.info(f"Std dev: {np.std(groups):,.0f} ({np.std(groups)/np.mean(groups)*100:.2f}% of mean)")
    
    # Summarize recommended model sizes for each group
    logging.info("\n=== Recommended Model Sizes for Each Group ===")
    
    def recommended_model_size(tokens):
        """Suggest a model size based on token count"""
        if tokens < 50_000_000:
            return "10-20M parameters (4 layers, 384d)"
        elif tokens < 100_000_000:
            return "40-70M parameters (6 layers, 512d)"
        elif tokens < 150_000_000:
            return "70-125M parameters (8 layers, 768d)"
        else:
            return "125-200M parameters (10-12 layers, 768-1024d)"
    
    # 3-group recommendations
    logging.info("3-group model size recommendations:")
    for i, tokens in enumerate([group1_tokens, group2_tokens, group3_tokens], 1):
        logging.info(f"Group {i}: {tokens:,} tokens → {recommended_model_size(tokens)}")
    
    # 4-group recommendations
    logging.info("\n4-group model size recommendations:")
    for i, tokens in enumerate([group1_tokens, group2_tokens, group3_tokens, group4_tokens], 1):
        logging.info(f"Group {i}: {tokens:,} tokens → {recommended_model_size(tokens)}")
    
    # 5-group recommendations
    logging.info("\n5-group model size recommendations:")
    for i, tokens in enumerate([group1_tokens, group2_tokens, group3_tokens, group4_tokens, group5_tokens], 1):
        logging.info(f"Group {i}: {tokens:,} tokens → {recommended_model_size(tokens)}")

if __name__ == "__main__":
    estimated_tokens = estimate_full_dataset_distribution()
    define_balanced_groups(estimated_tokens) 