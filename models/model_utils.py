from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, AutoModelForSeq2SeqLM, 
                          DataCollatorForLanguageModeling, DataCollatorForSeq2Seq)
from trl import DataCollatorForCompletionOnlyLM
import torch


class MixinModel():
    """
    Simple class to manage models more easily
    """
    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_collator(self):
        return self.data_collator
        
    def get_global_grad_norm(self, norm_type=2.0):
        """
        Get the gradient norm of all parameters, calculated as specified by norm_type.
        """
        parameters = [p for p in self.model.parameters() if p.grad is not None]
        
        if len(parameters) == 0:
            return torch.tensor(0.0)
        
        device = parameters[0].grad.device
        total_norm = torch.zeros([], device=device)
        
        for p in parameters:
            param_norm = p.grad.detach().data.norm(norm_type)
            total_norm += param_norm ** norm_type
            
        total_norm = total_norm ** (1.0 / norm_type)
        return total_norm
    

class GPT2Model(MixinModel):
    """
    GPT-2 model class
    """
    def __init__(self, checkpoint, device):
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, clean_up_tokenization_spaces=True)
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        #self.response_template = " ### Response: "
        #self.data_collator = DataCollatorForCompletionOnlyLM([44386, 18261, 25], tokenizer=self.tokenizer)
        #self.tokenizer.pad_token = self.tokenizer.eos_token #ONLY FOR GPT-2 Open generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.resize_token_embeddings(len(self.tokenizer))

    def tokenize_function(self, example):
        #text = example["prompt"] + "\n" + example["completion"] -> OLD WAY
        text = [p + c + self.tokenizer.eos_token for p, c in zip(example["prompt"], example["completion"])]
        input_encodings = self.tokenizer(text, truncation=True) #Maybe dont use truncation
        target_encodings = self.tokenizer(example["completion"], truncation=True)

        return {"input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "targets": target_encodings["input_ids"]} #because its necesary
        #return example
    
    
class T5Model(MixinModel):
    """
    T5 model class
    """
    def __init__(self, checkpoint, device):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, clean_up_tokenization_spaces=True)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)

    def tokenize_function(self, example):
        input_encodings = self.tokenizer(example["prompt"], truncation=True)
        target_encodings = self.tokenizer(text_target=example["completion"], truncation=True)

        return {"input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "labels": target_encodings["input_ids"]}
        
        
class PythiaModel(MixinModel):
    """
    Pythia model class
    """
    def __init__(self, checkpoint, device):
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, clean_up_tokenization_spaces=True)
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def tokenize_function(self, example):
        text = [p + c + self.tokenizer.eos_token for p, c in zip(example["prompt"], example["completion"])]
        input_encodings = self.tokenizer(text, truncation=True)
        target_encodings = self.tokenizer(example["completion"], truncation=True)

        return {"input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "targets": target_encodings["input_ids"]}


# TODO: Add more models here (SmolLM2Model, OPT, BlooM, etc.)
class SmolLMModel(MixinModel):
    """
    SmolLM model class
    """
    def __init__(self, checkpoint, device):
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, clean_up_tokenization_spaces=True)
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.resize_token_embeddings(len(self.tokenizer))

    def tokenize_function(self, example):
        text = [p + c + self.tokenizer.eos_token for p, c in zip(example["prompt"], example["completion"])]
        input_encodings = self.tokenizer(text, truncation=True)
        target_encodings = self.tokenizer(example["completion"], truncation=True)

        return {"input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "targets": target_encodings["input_ids"]}