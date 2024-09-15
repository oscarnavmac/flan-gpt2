from datasets import load_dataset
from data_utils import format_example
import random
import templates

commongen = load_dataset('allenai/common_gen', split='validation')

flan_pattern_name = "common_gen"
patterns_list = templates.PATTERNS[flan_pattern_name]

def tokenize_function(example):
    idx = random.randint(0, 9)
    raw_text = format_example(example, patterns_list, idx).values()
    text = ' '.join(raw_text)
    example["text"] = text
    return example

dataset = commongen.map(tokenize_function, batched=False)
print(dataset)
print(dataset[0]['text'])
print()
print(dataset[1]['text'])
print()
print(dataset[2]['text'])