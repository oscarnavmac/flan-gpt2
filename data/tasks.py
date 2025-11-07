from datasets import load_dataset, Dataset

# TODO: implement every dataset processor

# AUXILIARY FUNCTION
def enumerate_items(items_list):
    num_items = len(items_list)
    number_list = [str(i) for i in range(1, 1 + num_items)]
    numbered_items = [f"{num}. {item}" for num, item in zip(number_list, items_list)]
    numbered_items_str = '\n'.join(numbered_items)
    return numbered_items_str

# ============================== anli ==========================================
_repo_anli = 'facebook/anli'

def _process_anli(example, converter):
    example["answer"] = converter(example["label"])
    example["options"] = ["entailment", "neutral", "contradiction"]
    
    return example

def _load_anli(train=False):
    if train:
        dataset = load_dataset(_repo_anli, split='train_r1')
    else:
        dataset = load_dataset(_repo_anli, split='test_r1')
    int2str = dataset.features['label'].int2str
    dataset = dataset.map(_process_anli,
                          fn_kwargs={"converter": int2str},
                          keep_in_memory=True
                          )
    
    return dataset
    
    
# =============================== squad ========================================
_repo_squad = "rajpurkar/squad"
_SQUAD_MAX_LEN = 2000

def _process_squad(example):
    example["answer"] = example["answers"]["text"][0]
    
    return example

def _load_squad(train=False):
    if train:
        dataset = load_dataset(_repo_squad, split='train')
    else:
        dataset = load_dataset(_repo_squad, split='validation')
    dataset = dataset.filter(lambda example: len(example["context"]) <= _SQUAD_MAX_LEN, keep_in_memory=True)
    dataset = dataset.map(_process_squad, keep_in_memory=True)
    
    return dataset


# ============================== common_gen ====================================
_repo_common_gen = "allenai/common_gen"

def _process_common_gen():
    pass

def _load_common_gen(train=False):
    if train:
        dataset = load_dataset(_repo_common_gen, split='train')
    else:
        dataset = load_dataset(_repo_common_gen, split='validation')
    #dataset = dataset.map(_process_common_gen, keep_in_memory=True)
    
    return dataset
    

# ============================== cosmos_qa =====================================
_repo_cosmos_qa = "allenai/cosmos_qa"

def _process_cosmos_qa(example):
    answers = [
        example["answer0"],
        example["answer1"],
        example["answer2"],
        example["answer3"],
    ]
    example["options"] = answers
    answer_idx = example["label"]
    example["answer"] = answers[answer_idx]
    
    return example

def _load_cosmos_qa(train=False):
    if train:
        dataset = load_dataset(_repo_cosmos_qa, split='train', trust_remote_code=True)
    else:
        dataset = load_dataset(_repo_cosmos_qa, split='validation', trust_remote_code=True)
    dataset = dataset.map(_process_cosmos_qa, keep_in_memory=True)
    
    return dataset


# ================================= CoQA ======================================= (WARNING: Dataset contains less than 10k examples)
_repo_coqa = "stanfordnlp/coqa"
_COQA_MAX_LEN = 1600
def _process_coqa(example):
    example["numbered_questions"] = enumerate_items(example["questions"])
    example["numbered_answers"] = enumerate_items(example["answers"]["input_text"])
    
    return example

def _load_coqa(train=False):
    if train:
        dataset = load_dataset(_repo_coqa, split='train')
    else:
        dataset = load_dataset(_repo_coqa, split='validation')
    dataset = dataset.filter(lambda example: len(example["story"]) <= _COQA_MAX_LEN, keep_in_memory=True)
    dataset = dataset.map(_process_coqa, keep_in_memory=True)
    
    return dataset


# ================================ python_code ================================== (No templates required, already formatted in insructions)
#_repo_human_eval = "openai/openai_humaneval" # "nickrosh/Evol-Instruct-Code-80k-v1" # "leo009/python-programming-instructions"
_PYTHONCODE_MAX_LEN = 1500
_repo_python_code = "iamtarun/python_code_instructions_18k_alpaca"

def _process_python_code(example):
    if example["input"] not in ("Not applicable", ""):
        example["instruction"] += " " + example["input"]
    example["solution"] = "```python\n" + example["output"] + "\n```"
    
    return example

def _load_python_code(train=False):
    
    dataset = load_dataset(_repo_python_code, split='train')
    dataset_dict = dataset.train_test_split(test_size=0.1, shuffle=False) # Since dataset only has "train" split
    
    if train:
        dataset = dataset_dict["train"]
    else:
        dataset = dataset_dict["test"]
    dataset = dataset.filter(lambda example: len(example["output"]) <= _PYTHONCODE_MAX_LEN, keep_in_memory=True)
    dataset = dataset.map(_process_python_code, keep_in_memory=True)
    
    return dataset


# =================================== samsum =====================================
_repo_samsum = "knkarthick/samsum"
_SAMSUM_MAX_LEN = 1200

def _process_samsum(example):
    pass

def _load_samsum(train=False):
    if train:
        dataset = load_dataset(_repo_samsum, split='train')
    else:
        dataset = load_dataset(_repo_samsum, split='validation')
    #dataset = dataset.map(_process_samsum, keep_in_memory=True)
    dataset = dataset.filter(lambda example: len(example["dialogue"]) <= _SAMSUM_MAX_LEN, keep_in_memory=True)
    
    return dataset

# ============================== xsum ==========================================
_repo_xsum = "EdinburghNLP/xsum"
_XSUM_MAX_LEN = 2800 # 1500

def _process_xsum(example):
    pass

def _load_xsum(train=False):
    if train:
        dataset = load_dataset(_repo_xsum, split='train')
    else:
        dataset = load_dataset(_repo_xsum, split='validation')
    #dataset = dataset.map(_process_xsum, keep_in_memory=True)
    dataset = dataset.filter(lambda example: len(example["document"]) <= _XSUM_MAX_LEN, keep_in_memory=True)
    
    return dataset


# ============================== bool_q ==========================================
_repo_bool_q = 'google/boolq'
_COSMOSQA_MAX_LEN = 2000

def _process_bool_q(example):
    example["options"] = ["True", "False"]
    example["label"] = int(not example["answer"]) # invert label to match options
    
    return example

def _load_bool_q(train=False):
    if train:
        dataset = load_dataset(_repo_bool_q, split='train')
    else:
        dataset = load_dataset(_repo_bool_q, split='validation')
    dataset = dataset.filter(lambda example: len(example["passage"]) <= _COSMOSQA_MAX_LEN, keep_in_memory=True)
    dataset = dataset.map(_process_bool_q, keep_in_memory=True)
    
    return dataset


# =============================== paws ==========================================
#_repo_glue_mrpc = "SetFit/mrpc"
_repo_paws = "google-research-datasets/paws"

def _process_paws(example):
    options = ["No", "Yes"]
    example["options"] = options
    example["answer"] = options[example["label"]]
    
    return example

def _load_paws(train=False):
    if train:
        dataset = load_dataset(_repo_paws, "labeled_final", split="train")
    else:
        dataset = load_dataset(_repo_paws, "labeled_final", split="test")
    dataset = dataset.map(_process_paws, keep_in_memory=True)
    
    return dataset

# ================================ eng_spa ===================================== (NOT ON FLAN: consider manual tamplates)
_repo_eng_spa = "OscarNav/spa-eng"

def _process_eng_spa():
    pass

def _load_eng_spa(train=False):
    if train:
        dataset = load_dataset(_repo_eng_spa, split="train")
    else:
        dataset = load_dataset(_repo_eng_spa, split="test")
        #dataset.filter(lambda example: len(example["answer"].strip()) >= 10, keep_in_memory=True)
    #dataset = dataset.map(_process_eng_spa, keep_in_memory=True)
    
    return dataset

#--------------------------------------IN-CONTEXT EVALUATION----------------------------------


# ================================== quora ===================================== (NO CITE AVAILABLE) 
_repo_quora = "toughdata/quora-question-answer-dataset"
_QUORA_MAX_LEN = 100

def _process_quora(example):
    pass

def _load_quora(train=False):
    if train:
        dataset = load_dataset(_repo_quora, split="train")
    else:
        raise NotImplementedError
    dataset = dataset.filter(lambda example: len(example["answer"]) <= _QUORA_MAX_LEN, keep_in_memory=True)
    #dataset = dataset.map(_process_quora, keep_in_memory=True)
    
    return dataset
    
    
# ================================== alpaca =====================================
_repo_alpaca = "tatsu-lab/alpaca"
_ALPACA_MAX_LEN = 100

def _process_alpaca(example):
    if example["input"] != "":
        example["instruction"] += " " + example["input"]
        
    return example
        
def _load_alpaca(train=False):
    if train:
        dataset = load_dataset(_repo_alpaca, split="train")
    else:
        raise NotImplementedError
    dataset = dataset.filter(lambda example: len(example["output"]) <= _ALPACA_MAX_LEN, keep_in_memory=True)
    dataset = dataset.map(_process_alpaca, keep_in_memory=True)
    
    return dataset
    
#-----------------------------------------------------------------------------------------------------------------------------------
LOADERS = {
    'anli': _load_anli,
    'common_gen': _load_common_gen,
    'squad': _load_squad,
    'cosmos_qa': _load_cosmos_qa,
    'coqa': _load_coqa,
    'samsum': _load_samsum,
    'python_code': _load_python_code,
    'xsum': _load_xsum,
    'bool_q': _load_bool_q,
    'eng_spa': _load_eng_spa,
    'paws': _load_paws,
    'quora': _load_quora,
    'alpaca': _load_alpaca
}

################################### Main Class #################################

class TaskConfigs:
    #def __init__():
        #pass
    
    def load_task(task_name: str, train: bool):
        dataset = LOADERS.get(task_name, lambda _: "Invalid option")(train)
        if not isinstance(dataset, Dataset):
            raise KeyError(f"Unknown task '{task_name}'. Please choose from {list(LOADERS.keys())}")
        return dataset