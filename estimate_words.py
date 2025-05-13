import json
import numpy as np

# Token to word ratio is approximately 0.75 for English text with most tokenizers
# (tokens are generally smaller than words)
TOKEN_TO_WORD_RATIO = 0.75

# Analyze index.jsonl
paper_counts = {'total': 0}
token_lengths = []
tokens_per_category = {}

with open('temp_dataset/index.jsonl', 'r') as f:
    for line in f:
        paper_counts['total'] += 1
        entry = json.loads(line)
        cat = entry['cat']
        length = entry['length']
        
        token_lengths.append(length)
        
        if cat not in tokens_per_category:
            tokens_per_category[cat] = []
        tokens_per_category[cat].append(length)

# Calculate word estimates
token_lengths = np.array(token_lengths)
estimated_words = token_lengths * TOKEN_TO_WORD_RATIO
total_tokens = np.sum(token_lengths)
total_estimated_words = total_tokens * TOKEN_TO_WORD_RATIO

# Calculate per-category word estimates
category_word_stats = {}
for cat in tokens_per_category:
    cat_lengths = np.array(tokens_per_category[cat])
    cat_word_estimates = cat_lengths * TOKEN_TO_WORD_RATIO
    category_word_stats[cat] = {
        'count': len(cat_lengths),
        'avg_words': int(np.mean(cat_word_estimates)),
        'median_words': int(np.median(cat_word_estimates)),
        'total_words': int(np.sum(cat_word_estimates))
    }

# Print results
print(f'\nWord Estimate Summary:')
print(f'Total papers: {paper_counts["total"]}')
print(f'Total tokens: {total_tokens}')
print(f'Estimated total words: {int(total_estimated_words)}')
print(f'Avg words per paper: {int(np.mean(estimated_words))}')
print(f'Median words per paper: {int(np.median(estimated_words))}')

print(f'\nEstimated Word Length By Category:')
sorted_cats = sorted(category_word_stats.items(), key=lambda x: x[1]['avg_words'], reverse=True)
for cat, stats in sorted_cats:
    print(f'  - {cat}: {stats["count"]} papers, avg {stats["avg_words"]} words, total {stats["total_words"]} words') 