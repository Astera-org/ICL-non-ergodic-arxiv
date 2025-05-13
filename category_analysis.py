import json
from pathlib import Path
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def analyze_category_distribution(index_path, splits_path=None):
    """Analyze the token and paper count distribution across categories"""
    
    # Load index data
    logging.info(f"Loading index from {index_path}")
    categories = defaultdict(lambda: {'papers': 0, 'tokens': 0})
    
    with open(index_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            cat = entry['cat']
            length = entry['length']
            
            categories[cat]['papers'] += 1
            categories[cat]['tokens'] += length
    
    # Convert to DataFrame for easier analysis
    data = []
    for cat, stats in categories.items():
        data.append({
            'category': cat,
            'papers': stats['papers'],
            'tokens': stats['tokens'],
            'avg_tokens_per_paper': stats['tokens'] / stats['papers'] if stats['papers'] > 0 else 0
        })
    
    df = pd.DataFrame(data)
    df = df.sort_values('tokens', ascending=False)
    
    # Print statistics
    total_papers = df['papers'].sum()
    total_tokens = df['tokens'].sum()
    
    logging.info(f"Total: {total_papers:,} papers, {total_tokens:,} tokens")
    logging.info("\nCategory Distribution:")
    for _, row in df.iterrows():
        logging.info(f"{row['category']}: {row['papers']:,} papers, {row['tokens']:,} tokens, " +
                    f"{row['tokens']/total_tokens*100:.2f}% of total, " +
                    f"{int(row['avg_tokens_per_paper']):,} avg tokens/paper")
    
    if splits_path:
        analyze_splits(splits_path, df)
    
    return df

def analyze_splits(splits_path, category_df):
    """Analyze the distribution across train/val/test splits"""
    with open(splits_path, 'r') as f:
        splits = json.load(f)
    
    logging.info("\nSplit Distribution:")
    for split, papers in splits.items():
        logging.info(f"{split}: {len(papers):,} papers")
    
    return splits

def generate_balanced_groups(df, num_groups=3):
    """Generate semantically balanced groups with similar token counts"""
    # Define semantic groupings based on ArXiv categories
    semantic_groups = {
        # Mathematics
        'pure_math': ['math.AC', 'math.GR'],
        'applied_math': ['math.ST'],
        
        # Core Computer Science
        'algorithms': ['cs.DS'],
        'theory': ['cs.IT'],
        'systems': ['cs.SY', 'cs.CE'],
        
        # AI and Applications
        'ai': ['cs.AI', 'cs.NE'],
        'applications': ['cs.CV', 'cs.PL']
    }
    
    # Flatten the semantic groups and ensure all categories are included
    all_cats = set()
    for cats in semantic_groups.values():
        all_cats.update(cats)
    
    # Check for any missing categories
    df_cats = set(df['category'])
    if df_cats - all_cats:
        logging.warning(f"Categories not in any semantic group: {df_cats - all_cats}")
    if all_cats - df_cats:
        logging.warning(f"Semantic groups with no matching categories: {all_cats - df_cats}")
    
    # Strategy: First try to combine semantically related groups to reach target size
    total_tokens = df['tokens'].sum()
    target_tokens_per_group = total_tokens / num_groups
    
    logging.info(f"\nGenerating {num_groups} balanced groups")
    logging.info(f"Total tokens: {total_tokens:,}")
    logging.info(f"Target tokens per group: {target_tokens_per_group:,}")
    
    # Create groups based on different strategies
    strategies = {}
    
    # Strategy 1: Equal token counts with semantic grouping where possible
    token_counts = {}
    for cat, group in semantic_groups.items():
        matching_cats = [c for c in group if c in df_cats]
        if matching_cats:
            token_counts[cat] = df[df['category'].isin(matching_cats)]['tokens'].sum()
    
    sorted_groups = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Helper function to print groups
    def print_group(group_cats, name=""):
        cats_df = df[df['category'].isin(group_cats)]
        total_group_tokens = cats_df['tokens'].sum()
        total_group_papers = cats_df['papers'].sum()
        group_details = ", ".join([f"{cat}" for cat in sorted(group_cats)])
        logging.info(f"Group {name}: {group_details}")
        logging.info(f"  {total_group_papers:,} papers, {total_group_tokens:,} tokens, {total_group_tokens/total_tokens*100:.2f}% of total")
        return total_group_tokens
    
    # 3-group strategy
    logging.info("\n=== 3 Balanced Groups ===")
    groups_3 = []
    
    # Group 1: Math-focused (math.ST, math.GR, math.AC)
    group1 = ['math.ST', 'math.GR', 'math.AC']
    group1_tokens = print_group(group1, "1: Mathematics")
    groups_3.append({"name": "Mathematics", "categories": group1, "tokens": group1_tokens})
    
    # Group 2: Core CS (cs.DS, cs.IT, cs.SY, cs.CE)
    group2 = ['cs.DS', 'cs.IT', 'cs.SY', 'cs.CE']
    group2_tokens = print_group(group2, "2: Core Computer Science")
    groups_3.append({"name": "Core Computer Science", "categories": group2, "tokens": group2_tokens})
    
    # Group 3: AI & Applications (cs.AI, cs.NE, cs.CV, cs.PL)
    group3 = ['cs.AI', 'cs.NE', 'cs.CV', 'cs.PL']
    group3_tokens = print_group(group3, "3: AI & Applications")
    groups_3.append({"name": "AI & Applications", "categories": group3, "tokens": group3_tokens})
    
    # 4-group strategy
    logging.info("\n=== 4 Balanced Groups ===")
    groups_4 = []
    
    # Group 1: Pure Math (math.GR, math.AC)
    group1 = ['math.GR', 'math.AC']
    group1_tokens = print_group(group1, "1: Pure Mathematics")
    groups_4.append({"name": "Pure Mathematics", "categories": group1, "tokens": group1_tokens})
    
    # Group 2: Applied Math & Theory (math.ST, cs.IT)
    group2 = ['math.ST', 'cs.IT']
    group2_tokens = print_group(group2, "2: Applied Math & Information Theory")
    groups_4.append({"name": "Applied Math & Information Theory", "categories": group2, "tokens": group2_tokens})
    
    # Group 3: Algorithms & Systems (cs.DS, cs.SY, cs.CE)
    group3 = ['cs.DS', 'cs.SY', 'cs.CE']
    group3_tokens = print_group(group3, "3: Algorithms & Systems")
    groups_4.append({"name": "Algorithms & Systems", "categories": group3, "tokens": group3_tokens})
    
    # Group 4: AI & Applications (cs.AI, cs.NE, cs.CV, cs.PL)
    group4 = ['cs.AI', 'cs.NE', 'cs.CV', 'cs.PL']
    group4_tokens = print_group(group4, "4: AI & Applications")
    groups_4.append({"name": "AI & Applications", "categories": group4, "tokens": group4_tokens})
    
    # 5-group strategy
    logging.info("\n=== 5 Balanced Groups ===")
    groups_5 = []
    
    # Group 1: Pure Math (math.GR, math.AC)
    group1 = ['math.GR', 'math.AC']
    group1_tokens = print_group(group1, "1: Pure Mathematics")
    groups_5.append({"name": "Pure Mathematics", "categories": group1, "tokens": group1_tokens})
    
    # Group 2: Statistics (math.ST)
    group2 = ['math.ST']
    group2_tokens = print_group(group2, "2: Statistics")
    groups_5.append({"name": "Statistics", "categories": group2, "tokens": group2_tokens})
    
    # Group 3: Algorithms & Information Theory (cs.DS, cs.IT)
    group3 = ['cs.DS', 'cs.IT']
    group3_tokens = print_group(group3, "3: Algorithms & Information Theory")
    groups_5.append({"name": "Algorithms & Information Theory", "categories": group3, "tokens": group3_tokens})
    
    # Group 4: Systems & Engineering (cs.SY, cs.CE)
    group4 = ['cs.SY', 'cs.CE']
    group4_tokens = print_group(group4, "4: Systems & Engineering")
    groups_5.append({"name": "Systems & Engineering", "categories": group4, "tokens": group4_tokens})
    
    # Group 5: AI & Applications (cs.AI, cs.NE, cs.CV, cs.PL)
    group5 = ['cs.AI', 'cs.NE', 'cs.CV', 'cs.PL']
    group5_tokens = print_group(group5, "5: AI & Applications")
    groups_5.append({"name": "AI & Applications", "categories": group5, "tokens": group5_tokens})
    
    # Calculate and print statistics for each grouping strategy
    for i, groups in enumerate([groups_3, groups_4, groups_5], 3):
        tokens = [g["tokens"] for g in groups]
        logging.info(f"\n{i}-group statistics:")
        logging.info(f"Min tokens: {min(tokens):,}")
        logging.info(f"Max tokens: {max(tokens):,}")
        logging.info(f"Max/Min ratio: {max(tokens)/min(tokens):.2f}")
        logging.info(f"Std dev: {np.std(tokens):,.0f} ({np.std(tokens)/np.mean(tokens)*100:.2f}% of mean)")
    
    return {"groups_3": groups_3, "groups_4": groups_4, "groups_5": groups_5}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze ArXiv dataset category distribution")
    parser.add_argument("--index", type=Path, default=Path("./preprocessed_arxiv/index.jsonl"),
                       help="Path to index.jsonl file")
    parser.add_argument("--splits", type=Path, default=Path("./preprocessed_arxiv/splits.json"),
                       help="Path to splits.json file")
    parser.add_argument("--num_groups", type=int, default=3,
                       help="Number of balanced groups to generate")
    
    args = parser.parse_args()
    
    # Analyze category distribution
    df = analyze_category_distribution(args.index, args.splits)
    
    # Generate balanced groups
    generate_balanced_groups(df, args.num_groups) 