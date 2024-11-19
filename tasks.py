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

def _process_squad(example):
    example["answer"] = example["answers"]["text"][0]
    
    return example

def _load_squad(train=False):
    if train:
        dataset = load_dataset(_repo_squad, split='train')
    else:
        dataset = load_dataset(_repo_squad, split='validation')
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
        dataset = load_dataset(_repo_cosmos_qa, split='dev', trust_remote_code=True)
    dataset = dataset.map(_process_cosmos_qa, keep_in_memory=True)
    
    return dataset


# ================================= CoQA =======================================
_repo_coqa = "stanfordnlp/coqa"

def _process_coqa(example):
    example["numbered_questions"] = enumerate_items(example["questions"])
    example["numbered_answers"] = enumerate_items(example["answers"]["input_text"])
    
    return example

def _load_coqa(train=False):
    if train:
        dataset = load_dataset(_repo_coqa, split='train')
    else:
        dataset = load_dataset(_repo_coqa, split='validation')
    dataset = dataset.map(_process_coqa, keep_in_memory=True)
    
    return dataset


# ================================ human_eval ================================== (NOT ON FLAN: consider manual tamplates)
_repo_human_eval = "openai/openai_humaneval" # nickrosh/Evol-Instruct-Code-80k-v1

def _process_human_eval(example):
    example["canonical_solution"] = "```python\n" + example["canonical_solution"] + "\n```"
    
    return example

def _load_human_eval(train=False):
    if train:
        raise NotImplementedError
    else:
        dataset = load_dataset(_repo_human_eval, split='test')
    dataset = dataset.map(_process_human_eval, keep_in_memory=True)
    
    return dataset


# =================================== copa =====================================
#TODAVIA NO HAY

# ============================== xsum ==========================================
_repo_xsum = "EdinburghNLP/xsum"
_XSUM_MAX_LEN = 3000

def _process_xsum(example):
    if len(example["document"]) <= _XSUM_MAX_LEN:
    
        return example

def _load_xsum(train=False):
    if train:
        dataset = load_dataset(_repo_xsum, split='train')
    else:
        dataset = load_dataset(_repo_xsum, split='validation')
    dataset = dataset.filter(_process_xsum, keep_in_memory=True) # This is used as a filter rather than a map
    
    return dataset


# ============================== bool_q ==========================================
_repo_bool_q = 'google/boolq'

def _process_bool_q(example):
    example["options"] = ["True", "False"]
    example["label"] = int(example["answer"])
    
    return example

def _load_bool_q(train=False):
    if train:
        dataset = load_dataset(_repo_bool_q, split='train')
    else:
        dataset = load_dataset(_repo_bool_q, split='validation')
    dataset = dataset.map(_process_bool_q, keep_in_memory=True)
    
    return dataset


# =============================== glue_mrpc ====================================
_repo_glue_mrpc = "SetFit/mrpc"

# ================================ eng-spa ===================================== (NOT ON FLAN: consider manual tamplates)
_repo_eng_spa = "OscarNav/spa-eng"

#-------------------------------------------------------------------------------
LOADERS = {
    'anli': _load_anli,
    'common_gen': _load_common_gen,
    'squad': _load_squad,
    'cosmos_qa': _load_cosmos_qa,
    'coqa': _load_coqa,
    'human_eval': _load_human_eval,
    'xsum': _load_xsum,
    'bool_q': _load_bool_q
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