import json
import numpy as np

# Load splits data
with open('temp_dataset/splits.json', 'r') as f:
    splits = json.load(f)

# Create dictionary to store paper details
paper_details = {}
with open('temp_dataset/index.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line)
        paper_details[entry['paper_id']] = {
            'category': entry['cat'],
            'tokens': entry['length']
        }

# Initialize data structure to track tokens by category and split
tokens_by_category_and_split = {}
for split_name in splits.keys():
    tokens_by_category_and_split[split_name] = {}

# Compute tokens for each category and split
for split_name, paper_ids in splits.items():
    split_total = 0
    
    # Process each paper in this split
    for paper_id in paper_ids:
        paper = paper_details.get(paper_id)
        if not paper:
            continue
            
        category = paper['category']
        tokens = paper['tokens']
        
        # Add category if not already tracked in this split
        if category not in tokens_by_category_and_split[split_name]:
            tokens_by_category_and_split[split_name][category] = {
                'paper_count': 0,
                'total_tokens': 0
            }
        
        # Update counts
        tokens_by_category_and_split[split_name][category]['paper_count'] += 1
        tokens_by_category_and_split[split_name][category]['total_tokens'] += tokens
        split_total += tokens
    
    # Add total to the data
    tokens_by_category_and_split[split_name]['total'] = split_total

# Print results
all_categories = set()
for split_data in tokens_by_category_and_split.values():
    for category in split_data.keys():
        if category != 'total':
            all_categories.add(category)

# Calculate grand total of tokens
grand_total = sum(split_data['total'] for split_data in tokens_by_category_and_split.values())

print(f"Tokens by Category and Split (with Word Estimates):")
print(f"===================================================\n")

# Print summary
print(f"GRAND TOTAL: {grand_total:,} tokens (~{int(grand_total * 0.75):,} words)\n")
for split_name in sorted(tokens_by_category_and_split.keys()):
    split_data = tokens_by_category_and_split[split_name]
    print(f"{split_name.upper()} TOTAL: {split_data['total']:,} tokens (~{int(split_data['total'] * 0.75):,} words)")

# Print detailed breakdown for each split
for split_name in sorted(tokens_by_category_and_split.keys()):
    split_data = tokens_by_category_and_split[split_name]
    
    print(f"\n{split_name.upper()} BY CATEGORY:")
    print(f"{'Category':<10} {'Papers':<8} {'Tokens':<12} {'Words (est.)':<12} {'% of Split':<12}")
    print(f"{'-'*55}")
    
    # Sort categories by token count
    categories_sorted = sorted(
        [(cat, data) for cat, data in split_data.items() if cat != 'total'],
        key=lambda x: x[1]['total_tokens'],
        reverse=True
    )
    
    for category, data in categories_sorted:
        token_count = data['total_tokens']
        paper_count = data['paper_count']
        percent_of_split = (token_count / split_data['total']) * 100
        
        print(f"{category:<10} {paper_count:<8} {token_count:<12,} {int(token_count * 0.75):<12,} {percent_of_split:<12.2f}%")

# Create a table showing percentage of tokens per category across splits
print(f"\nPERCENTAGE OF TOKENS BY CATEGORY ACROSS SPLITS:")
print(f"{'Category':<10}", end="")
for split_name in sorted(tokens_by_category_and_split.keys()):
    print(f"{split_name.upper():<12}", end="")
print()

print(f"{'-'*46}")

sorted_categories = sorted(all_categories)
for category in sorted_categories:
    print(f"{category:<10}", end="")
    for split_name in sorted(tokens_by_category_and_split.keys()):
        split_data = tokens_by_category_and_split[split_name]
        if category in split_data:
            token_count = split_data[category]['total_tokens']
            percent_of_split = (token_count / split_data['total']) * 100
            print(f"{percent_of_split:<12.2f}%", end="")
        else:
            print(f"{'0.00%':<12}", end="")
    print() 