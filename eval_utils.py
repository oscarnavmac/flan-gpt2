from metrics import METRIC
from tasks import TaskConfigs
from data_utils import format_instructions
import templates
from tqdm import tqdm
import re
import torch
import warnings

def convert(text, options):
    if text in options:
        return options.index(text)
    else:
        return -1

def get_label(generated, options):
    word = generated.partition(' ')[0] #.partition(' ')[0] for picking only the first word
    cleaned_word = re.sub(r'[^\w\s]', '', word.lstrip()) # Dont discard generations such as entailment:, True?, etc...
    label = convert(cleaned_word, options)
    return label

class Evaluation:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def generate(self, input_list, return_full_text=True, max_tokens=60):
        outputs = []
        for input in tqdm(input_list, desc="Generating responses... "):
            #input += " ### Response: "
            inputs = self.tokenizer(input, return_tensors='pt').to(self.device)
            input_length = len(self.tokenizer.decode(inputs["input_ids"][0]))
            with torch.no_grad():
                output = self.tokenizer.decode(
                    self.model.generate(
                        inputs["input_ids"],
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                        max_new_tokens=max_tokens,
                        do_sample=True
                    )[0],
                    skip_special_tokens=True
                )
            if return_full_text: 
                outputs.append(output) 
            else: 
                outputs.append(output[input_length:].strip())

        return outputs
    
    
    def rank_classification(self, inputs_list, options_list):
        """" Rank Classification """
        outputs = []
        for prompt, options in tqdm(zip(inputs_list, options_list), desc="Generating predictions... "):
            choice_probs = []
            for completion in options:
                inputs = self.tokenizer(prompt + completion, return_tensors='pt').to(self.device)
                with torch.no_grad():
                    output = self.model(**inputs, labels=inputs["input_ids"])
                    # Get the negative log-likelihood as a score for this choice
                    choice_prob = -output.loss.item()
                choice_probs.append(choice_prob)
            # Select the choice with the highest probability
            predicted_answer = torch.argmax(torch.tensor(choice_probs)).item()
            outputs.append(predicted_answer)
            
        return outputs
    
    
    def evaluate(self, dataset_name, num_samples=None, training_set=False, return_full_text=True):
        
        loaded = TaskConfigs.load_task(dataset_name, training_set).filter(
            lambda example, idx: idx < num_samples, with_indices=True)
        patterns = templates.PATTERNS[dataset_name][:1] # Only first template for any task
        
        dataset = loaded.map(format_instructions,
                            #load_from_cache_file=False,
                            batched=False,
                            fn_kwargs={"patterns_list": patterns})
        input_list = dataset["prompt"]
        
        if 'options' in dataset[0]: # rank classification
            references = dataset["label"]
            options = dataset["options"]
            predictions = self.rank_classification(inputs_list=input_list, options_list=options)
        else:
            references = dataset["completion"]
            predictions = self.generate(input_list, return_full_text)
            
        metric_fn = METRIC[dataset_name]
        result = list(metric_fn(references, predictions).values()) # Get value of the only element in the dict
        
        return float(result[0])
    
    
    def evaluate_dataset(self, dataset_name, num_samples=None, training_set=False, return_full_text=True):
        warnings.warn("deprecated", DeprecationWarning)
        
        loaded = TaskConfigs.load_task(dataset_name, training_set).filter(
            lambda example, idx: idx < num_samples, with_indices=True)
        patterns = templates.PATTERNS[dataset_name][:1] # Only first template for any task
        
        dataset = loaded.map(format_instructions,
                            #load_from_cache_file=False,
                            batched=False,
                            fn_kwargs={"patterns_list": patterns})
        
        references = dataset["completion"]
        input_list = dataset["prompt"]
        predictions = self.generate(input_list, return_full_text)
        metric_fn = METRIC[dataset_name]

        if 'options' in dataset[0]:
            options = dataset["options"]
            references = [get_label(r, o) for r, o in zip(references, options)]
            predictions = [get_label(p, o) for p, o in zip(predictions, options)]

        result = list(metric_fn(references, predictions).values()) # Get value of the only element in the dict
        
        return float(result[0])