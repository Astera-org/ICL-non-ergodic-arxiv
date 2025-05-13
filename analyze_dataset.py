import json
import numpy as np

# Analyze index.jsonl
paper_counts = {'total': 0}
categories = set()
token_lengths = []
tokens_per_category = {}

with open('temp_dataset/index.jsonl', 'r') as f:
    for line in f:
        paper_counts['total'] += 1
        entry = json.loads(line)
        cat = entry['cat']
        length = entry['length']
        
        categories.add(cat)
        token_lengths.append(length)
        
        if cat not in tokens_per_category:
            tokens_per_category[cat] = []
        tokens_per_category[cat].append(length)

# Analyze splits.json
with open('temp_dataset/splits.json', 'r') as f:
    splits = json.load(f)
    for split in splits:
        paper_counts[split] = len(splits[split])

# Calculate statistics
token_lengths = np.array(token_lengths)
category_stats = {}
for cat in tokens_per_category:
    cat_lengths = np.array(tokens_per_category[cat])
    category_stats[cat] = {
        'count': len(cat_lengths),
        'min_tokens': int(np.min(cat_lengths)),
        'max_tokens': int(np.max(cat_lengths)),
        'avg_tokens': int(np.mean(cat_lengths)),
        'median_tokens': int(np.median(cat_lengths)),
        'total_tokens': int(np.sum(cat_lengths))
    }

# Print results
print(f'\nDataset Summary:')
print(f'Total papers: {paper_counts["total"]}')
print(f'Categories: {len(categories)}')
print(f'Split distribution:')
for split in splits:
    print(f'  - {split}: {paper_counts[split]} papers')

print(f'\nToken Statistics:')
print(f'Min tokens per paper: {np.min(token_lengths)}')
print(f'Max tokens per paper: {np.max(token_lengths)}')
print(f'Avg tokens per paper: {int(np.mean(token_lengths))}')
print(f'Median tokens per paper: {int(np.median(token_lengths))}')
print(f'Total tokens: {np.sum(token_lengths)}')

print(f'\nCategory Distribution:')
sorted_cats = sorted(category_stats.items(), key=lambda x: x[1]['count'], reverse=True)
for cat, stats in sorted_cats:
    print(f'  - {cat}: {stats["count"]} papers, avg {stats["avg_tokens"]} tokens')

print(f'\nDetailed Category Stats:')
for cat, stats in sorted_cats:
    print(f'  {cat}: {stats["count"]} papers, {stats["total_tokens"]} tokens total')
    print(f'    min: {stats["min_tokens"]}, max: {stats["max_tokens"]}, avg: {stats["avg_tokens"]}, median: {stats["median_tokens"]}') 