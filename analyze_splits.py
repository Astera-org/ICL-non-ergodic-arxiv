import json
import numpy as np

# Analyze splits.json and index.jsonl
with open('temp_dataset/splits.json', 'r') as f:
    splits = json.load(f)

# Create mapping from paper_id to category
paper_id_to_cat = {}
with open('temp_dataset/index.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line)
        paper_id_to_cat[entry['paper_id']] = entry['cat']

# Count categories per split
categories_per_split = {}
for split_name, paper_ids in splits.items():
    if split_name not in categories_per_split:
        categories_per_split[split_name] = {}
    
    for paper_id in paper_ids:
        cat = paper_id_to_cat.get(paper_id)
        if cat:
            if cat not in categories_per_split[split_name]:
                categories_per_split[split_name][cat] = 0
            categories_per_split[split_name][cat] += 1

# Calculate percentages
split_percentages = {}
for split_name, cat_counts in categories_per_split.items():
    total = sum(cat_counts.values())
    split_percentages[split_name] = {
        cat: {
            'count': count,
            'percentage': round(100 * count / total, 2)
        }
        for cat, count in cat_counts.items()
    }

# Print results
print("\nCategory Distribution Across Splits:")
for split_name in sorted(split_percentages.keys()):
    total_papers = sum(cat_info['count'] for cat_info in split_percentages[split_name].values())
    print(f"\n{split_name.upper()} ({total_papers} papers):")
    
    # Sort by count
    sorted_cats = sorted(
        split_percentages[split_name].items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    
    for cat, info in sorted_cats:
        print(f"  - {cat}: {info['count']} papers ({info['percentage']}%)")

# Calculate total number of tokens per split
total_tokens_per_split = {}
for split_name, paper_ids in splits.items():
    total_tokens_per_split[split_name] = 0
    
    for paper_id in paper_ids:
        for line in open('temp_dataset/index.jsonl', 'r'):
            entry = json.loads(line)
            if entry['paper_id'] == paper_id:
                total_tokens_per_split[split_name] += entry['length']
                break

print("\nTokens per Split:")
for split_name, total_tokens in total_tokens_per_split.items():
    print(f"  - {split_name}: {total_tokens} tokens (~{int(total_tokens * 0.75)} words)") 