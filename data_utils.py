from datasets import concatenate_datasets
from tasks import TaskConfigs
import random
import templates

MAX_NUM_EXAMPLES = 5000
TASKS = {
    "common_gen": "allenai/common_gen",
    "xsum": "EdinburghNLP/xsum",
    "bool_q": "google/boolq",
    "anli": "facebook/anli"
}

def format_options(example):
    """Formats options (if any) for FLAN tasks."""
    example['options_'] = 'OPTIONS:\n- ' + '\n- '.join(example['options'])
    return example


def format_example(example, patterns_list, i):
    """Converts every example into a instruction format according to a pattern template"""
    inputs_pattern, targets_pattern = patterns_list[i]
    format_strings = {'prompt': inputs_pattern, 'completion': targets_pattern}
    new_example = {}
    for f_name, format_str in format_strings.items():
        new_example[f_name] = format_str.format(**example)
        
    return new_example


def format_instructions(example, patterns_list):
    idx = random.randint(0, len(patterns_list)-1)
    if 'options' in example:
        example = format_options(example)
    example = format_example(example, patterns_list, idx)
    
    return example


def create_instruct_dataset(tasks_list, training_set=True): #check
    datasets = []
    for name in tasks_list:
        loaded = TaskConfigs.load_task(name, training_set).filter(
            lambda example, idx: idx < MAX_NUM_EXAMPLES, with_indices=True)
        patterns = templates.PATTERNS[name]
        dataset = loaded.map(format_instructions,
                            #load_from_cache_file=False,
                            batched=False,
                            fn_kwargs={"patterns_list": patterns},
                            remove_columns=loaded.column_names)
        if 'options_' in dataset[0]:
            dataset = dataset.remove_columns(["options_"])
        datasets.append(dataset)

    return concatenate_datasets(datasets).shuffle()