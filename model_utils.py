from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, AutoModelForSeq2SeqLM, 
                          DataCollatorForLanguageModeling, DataCollatorForSeq2Seq)
import torch

class GPT2Model():
    """
    Simple class to manage GPT-2 models more easily
    """
    def __init__(self, checkpoint, device):
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.tokenizer.pad_token = self.tokenizer.eos_token #ONLY FOR GPT-2
    
    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_collator(self):
        return self.data_collator

    def tokenize_function(self, example):
        text = example["prompt"] + "\n" + example["completion"]
        input_encodings = self.tokenizer(text, truncation=True)
        length = len(input_encodings.input_ids)
        target_encodings = self.tokenizer(example["completion"], padding='max_length', max_length=length)

        return {"input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "targets": target_encodings["input_ids"]}
        #return example
    
class T5Model():
    """
    Simple class to manage T5 models more easily
    """
    def __init__(self, checkpoint, device):
        self.model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.data_collator = DataCollatorForSeq2Seq(self.tokenizer)
    
    def get_model(self):
        return self.model

    def get_tokenizer(self):
        return self.tokenizer
    
    def get_collator(self):
        return self.data_collator

    def tokenize_function(self, example):
        input_encodings = self.tokenizer(example["prompt"], truncation=True)

        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(example["completion"], truncation=True)

        return {"input_ids": input_encodings["input_ids"],
                "attention_mask": input_encodings["attention_mask"],
                "labels": target_encodings["input_ids"]}