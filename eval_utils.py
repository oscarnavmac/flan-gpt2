import evaluate
from datasets import load_dataset
from data_utils import format_example
from templates import PATTERNS
from tqdm import tqdm

def generate(model, tokenizer, device, input_list, tagging):
    outputs = []
    for input in tqdm(input_list, desc="Evaluating model"):
        inputs = tokenizer(input, return_tensors='pt').to(device)
        input_length = len(tokenizer.decode(inputs["input_ids"][0]))
        output = tokenizer.decode(
            model.generate(
                inputs["input_ids"],
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=10,
                do_sample=True
            )[0],
            skip_special_tokens=True
        )[input_length:].strip()#.partition(' ')[0]
        outputs.append(output)

    return outputs

def get_label(generated, tagging):
    word = generated.partition(' ')[0] #.partition(' ')[0] for picking only the first word
    try:
        label = tagging(word)
    except:
        label = -1
    #print(word)

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

class Evaluation:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def anli(self):
        dataset = load_dataset("facebook/anli", split="test_r1")
        int2str = dataset.features['label'].int2str
        str2int = dataset.features['label'].str2int
        dataset = dataset.map(lambda example: {"answer": int2str(example["label"])})
        references = dataset["label"]
        index = 0 # We will be using the first template for any task
        input_list = [format_example(ex, PATTERNS["anli"], index)["prompt"] for ex in dataset]
        predictions = generate_labels(self.model, self.tokenizer, self.device, input_list[:2], str2int)
        result = accuracy(references[:2], predictions)["accuracy"]
        print(f"Total accuracy is {result}")

    def bool_q(self):
        dataset = load_dataset('google/boolq', split='validation')
        references = [int(ans) for ans in dataset["answer"]]
        index = 0 # We will be using the first template for any task
        input_list = [format_example(ex, PATTERNS["bool_q"], index)["prompt"] for ex in dataset]
        predictions = generate_labels(self.model, self.tokenizer, self.device, input_list[:2], str2bool)
        result = accuracy(references[:2], predictions)["accuracy"]
        print(f"Total accuracy is {result}")

    def common_gen():
        pass