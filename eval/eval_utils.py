from eval.metrics import METRIC
from data.tasks import TaskConfigs
from data.data_utils import format_instructions
import data.templates as templates
from tqdm import tqdm
import re
import torch


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
        
        self.model.eval()

    def generate(self, prompts_list, return_full_text=True, max_tokens=200, temperature=1.0):
        outputs = []
        for input in tqdm(prompts_list, desc="Generating responses... "):
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
                        do_sample=True,
                        temperature=temperature,
                    )[0],
                    skip_special_tokens=True
                )
            if return_full_text: 
                outputs.append(output) 
            else: 
                outputs.append(output[input_length:].strip())

        return outputs
    
    def rank_classification(self, prompts_list, options_list):
        """Rank Classification: select the option with highest conditional log-likelihood."""
        outputs = []

        for prompt, options in tqdm(
            zip(prompts_list, options_list),
            desc="Generating predictions... ",
            total=len(prompts_list)
        ):
            choice_scores = []
            for option in options:
                # Tokenize prompt and option separately
                inputs = self.tokenizer(prompt, option, return_tensors='pt').to(self.device)
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                # Mask prompt tokens so that only the option contributes to loss
                # tokenizer(prompt, option) returns tokens like [BOS] prompt tokens + option tokens
                with torch.no_grad():
                    labels = input_ids.clone()
                    prompt_len = self.tokenizer(prompt, return_tensors='pt')["input_ids"].size(1)
                    labels[:, :prompt_len] = -100  # ignore prompt in loss computation

                    output = self.model(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        labels=labels)
                    choice_score = -output.loss.item()  # higher = better

                choice_scores.append(choice_score)

            best_idx = max(range(len(choice_scores)), key=lambda i: choice_scores[i])
            outputs.append(best_idx)

        return outputs

    
    def build_prompt_list(self, dataset, n_shot=0):
        """ Build the prompt list for the model using first n_shot examples of the dataset """
        prompt_list = dataset["prompt"]
        if n_shot > 0:
            shot_prompts, prompt_list = prompt_list[:n_shot], prompt_list[n_shot:]
            shot_examples = [shot_prompts[i] + " " + dataset["completion"][i] for i in range(n_shot)]
            for i in range(len(prompt_list)):
                prompt_list[i] = "\n\n".join(shot_examples) + "\n\n" + prompt_list[i]
        
        return prompt_list

    def evaluate(self, dataset_name, num_samples=None, n_shot=0, training_set=False, return_full_text=True, 
                 max_tokens=200, temperature=1.0, shuffle=False, verbose=False) -> dict[str, float]:
        """ Evaluate a dataset with the model """
        loaded = TaskConfigs.load_task(dataset_name, training_set).filter(
            lambda example, idx: idx < num_samples, with_indices=True)
        patterns = templates.PATTERNS[dataset_name][:1] # Only first template for any task
        
        dataset = loaded.map(format_instructions,
                            #load_from_cache_file=False,
                            batched=False,
                            fn_kwargs={"patterns_list": patterns})
        if shuffle:
            dataset = dataset.shuffle(seed=42)
        prompts_list = self.build_prompt_list(dataset, n_shot=n_shot)
        if verbose:
            print("Prompts:")
            for p in prompts_list:
                print(p)
            print("\n\n\n")
        
        if 'options' in dataset[0]: # apply rank classification
            references = dataset["label"][n_shot:]
            options = dataset["options"]
            predictions = self.rank_classification(prompts_list=prompts_list, options_list=options)
        else:
            references = dataset["completion"][n_shot:]
            predictions = self.generate(prompts_list, return_full_text=return_full_text, 
                                        max_tokens=max_tokens, temperature=temperature)
            
        if verbose:
            print("References:")
            for r in references:
                print(r)
            print("\n\n\n")
            print("Predictions:")
            for p in predictions:
                print(p)
            print("\n\n\n")
            
        metrics = METRIC[dataset_name]
        
        results = {}
        for metric_fn in metrics:
            metric_name = metric_fn.__name__
            if dataset_name == "common_gen" and metric_name == "coverage":
                result = metric_fn(dataset["concepts"][n_shot:], predictions)
            elif dataset_name == "squad" and metric_name == "squad":
                squad_references = [{"id": dataset["id"][i], "answers": dataset["answers"][i]} for i in range(n_shot, len(dataset["id"]))]
                squad_predictions = [{"id": dataset["id"][i], "prediction_text": pred} for i, pred in enumerate(predictions, start=n_shot)]
                result = metric_fn(squad_references, squad_predictions)
            else:
                result = metric_fn(references, predictions)

            score = float(result["score"]) * 100
            results[metric_name] = score

            if verbose:
                print(f"{dataset_name} ({metric_fn.__name__}) score: {score}")
                
        return results
        
        # result = list(metric_fn(references, predictions).values()) # Get value of the only element in the dict
        
        # return float(result[0])