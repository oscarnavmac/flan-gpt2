from datasets import load_dataset, concatenate_datasets
import random
import templates

MAX_NUM_EXAMPLES = 10000
TASKS = {
    "common_gen": "allenai/common_gen",
    "xsum": "EdinburghNLP/xsum",
    "bool_q": "google/boolq",
    "anli": "facebook/anli"
}

def format_example(example, patterns_list, i):
    inputs_pattern, targets_pattern = patterns_list[i]
    format_strings = {'prompt': inputs_pattern, 'completion': targets_pattern}
    new_example = {}
    for f_name, format_str in format_strings.items():
        new_example[f_name] = format_str.format(**example)
    return new_example

def preprocess_function(example, patterns_list):
    idx = random.randint(0, 9)
    raw_text = format_example(example, patterns_list, idx).values()
    example["text"] = ' '.join(raw_text)
    return example

def create_instruct_dataset(tasks_list):
    datasets = []
    for name in tasks_list:
        if name=="anli":
            loaded = load_dataset(TASKS[name], split='train_r1').filter(
                lambda example, idx: idx < MAX_NUM_EXAMPLES, with_indices=True)
            int2str = loaded.features['label'].int2str
            loaded = loaded.map(lambda example: {"answer": int2str(example["label"])})
        else:
            loaded = load_dataset(TASKS[name], split='train').filter(
                lambda example, idx: idx < MAX_NUM_EXAMPLES, with_indices=True)
        patterns = templates.PATTERNS[name]
        dataset = loaded.map(preprocess_function, 
                            batched=False,
                            fn_kwargs={"patterns_list": patterns},
                            remove_columns=loaded.column_names,)
        datasets.append(dataset)

    return concatenate_datasets(datasets).shuffle()