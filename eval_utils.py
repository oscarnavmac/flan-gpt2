import evaluate
from datasets import load_dataset
from data_utils import format_example
from templates import PATTERNS
from tqdm import tqdm
import re

def generate(model, tokenizer, device, input_list):
    outputs = []
    for input in tqdm(input_list, desc="Evaluating model"):
        inputs = tokenizer(input, truncation=True, return_tensors='pt').to(device)
        input_length = len(tokenizer.decode(inputs["input_ids"][0]))
        output = tokenizer.decode(
            model.generate(
                inputs["input_ids"],
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=40,
                do_sample=True
            )[0],
            skip_special_tokens=True
        )[input_length:].strip()
        outputs.append(output)

    return outputs

def get_label(generated, tagging):
    word = generated.partition(' ')[0] #.partition(' ')[0] for picking only the first word
    cleaned_word = re.sub(r'[^\w\s]', '', word) # Dont discard generations such as entailment:, True?, etc...
    try:
        label = tagging(cleaned_word)
    except:
        label = -1
    #print(cleaned_word)
    return label

def str2bool(str):
    if str.lower() == "true":
        return True
    elif str.lower() == "false":
        return False
    else: raise Exception("Not a Boolean")

def accuracy(references, predictions):
    acc_metric = evaluate.load("accuracy")
    return acc_metric.compute(references=references, predictions=predictions)

def rouge(references, predictions):
    rouge_metric = evaluate.load('rouge')
    return rouge_metric.compute(predictions=predictions,
                  references=references,
                  use_aggregator=True)

class Evaluation:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def anli(self, max_examples=None):
        dataset = load_dataset("facebook/anli", split="test_r1")
        if max_examples is not None:
            dataset = dataset.filter(
                lambda example, idx: idx < max_examples, with_indices=True)
        int2str = dataset.features['label'].int2str
        str2int = dataset.features['label'].str2int
        dataset = dataset.map(lambda example: {"answer": int2str(example["label"])})
        references = dataset["label"]
        index = 0 # We will be using the first template for any task
        input_list = [format_example(ex, PATTERNS["anli"], index)["prompt"] for ex in dataset]
        predictions = generate(self.model, self.tokenizer, self.device, input_list)
        predictions = [get_label(pred, str2int) for pred in predictions]
        result = accuracy(references, predictions)["accuracy"]
        return result

    def bool_q(self, max_examples=None):
        dataset = load_dataset('google/boolq', split='validation')
        if max_examples is not None:
            dataset = dataset.filter(
                lambda example, idx: idx < max_examples, with_indices=True)
        references = [int(ans) for ans in dataset["answer"]]
        index = 0 # We will be using the first template for any task
        input_list = [format_example(ex, PATTERNS["bool_q"], index)["prompt"] for ex in dataset]
        predictions = generate(self.model, self.tokenizer, self.device, input_list)
        predictions = [get_label(pred, str2bool) for pred in predictions]
        result = accuracy(references, predictions)["accuracy"]
        return result

    def common_gen(self, max_examples=None):
        dataset = load_dataset('allenai/common_gen', split='validation')
        if max_examples is not None:
            dataset = dataset.filter(
                lambda example, idx: idx < max_examples, with_indices=True)
        references = dataset["target"]
        index = 0 # We will be using the first template for any task
        input_list = [format_example(ex, PATTERNS["common_gen"], index)["prompt"] for ex in dataset]
        predictions = generate(self.model, self.tokenizer, self.device, input_list)
        result = rouge(references, predictions)["rouge1"]
        return result

    def xsum(self, max_examples=None):
        dataset = load_dataset('EdinburghNLP/xsum', split='test')
        if max_examples is not None:
            dataset = dataset.filter(
                lambda example, idx: idx < max_examples, with_indices=True)
        references = dataset["summary"]
        index = 0 # We will be using the first template for any task
        input_list = [format_example(ex, PATTERNS["xsum"], index)["prompt"] for ex in dataset]
        predictions = generate(self.model, self.tokenizer, self.device, input_list)
        result = rouge(references, predictions)["rougeLsum"]
        return result