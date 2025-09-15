from transformers import (AutoTokenizer, 
                          AutoModelForCausalLM, AutoModelForSeq2SeqLM, 
                          DataCollatorForLanguageModeling, DataCollatorForSeq2Seq)
#from trl import DataCollatorForCompletionOnlyLM
import torch
from peft import LoraConfig, TaskType, PeftModel, PeftConfig
from transformers import BitsAndBytesConfig



class MixinModel():
    """
    Simple class to manage models more easily
    """
    def __init__(self, checkpoint: str, device: str, peft: bool = False, quantization: bool = False):
        """
        Initialize the model, tokenizer, and data collator.
        """
        # None values will be set in child classes
        self.checkpoint = checkpoint
        self.device = device
        self.peft = peft
        self.model = None
        self.tokenizer = None
        self.data_collator = None
        self.task_type = None
        self.target_modules = None
        
        self.model_path = checkpoint
        
        if self.peft:
            try:
                peft_config = PeftConfig.from_pretrained(self.checkpoint)
                self.model_path = peft_config.base_model_name_or_path
            except ValueError:
                print("No PEFT config found, using base checkpoint")
                pass
        
    def get_model(self, peft=False, lora_params=None):
        """Get the model, optionally with PEFT or LoRA applied."""
        if not peft:
            return self.model

        # In evaluation mode, lora_params should be None
        if lora_params is None:
            return PeftModel.from_pretrained(self.model, self.checkpoint)
        
        lora_config = LoraConfig(
            r=lora_params["rank"],  # LoRA rank
            lora_alpha=lora_params["alpha"],  # LoRA alpha parameter
            lora_dropout=lora_params["dropout"],  # LoRA dropout rate
            bias=lora_params["bias"],  # Bias handling
            task_type=self.task_type,
            target_modules=self.target_modules
        )
        self.model.add_adapter(lora_config, adapter_name="lora_1")
        return self.model#get_peft_model(self.model, lora_config)

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
    
    def print_trainable_parameters(self):
            """
            Print the number of trainable parameters in the model.
            """
            trainable_params = 0
            all_param = 0
            for _, param in self.model.named_parameters():
                all_param += param.numel()
                if param.requires_grad:
                    trainable_params += param.numel()
            print(
                f"Trainable params: {trainable_params} || "
                f"All params: {all_param} || "
                f"Trainable: {100 * trainable_params / all_param}%"
            )
    

class GPT2Model(MixinModel):
    """
    GPT-2 model class
    """
    def __init__(self, checkpoint: str, device: str, peft: bool = False):
        super().__init__(checkpoint, device, peft)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, clean_up_tokenization_spaces=True)
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.task_type = TaskType.CAUSAL_LM
        self.target_modules = ["c_attn", "c_proj"]  # Specific to GPT2 models
        #self.response_template = " ### Response: "
        #self.data_collator = DataCollatorForCompletionOnlyLM([44386, 18261, 25], tokenizer=self.tokenizer)
        #self.tokenizer.pad_token = self.tokenizer.eos_token #ONLY FOR GPT-2 Open generation
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<pad>"})
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.resize_token_embeddings(len(self.tokenizer))

    def tokenize_function(self, example):
        # Handle both single examples and batched examples
        if isinstance(example['prompt'], list):
            # Batched processing
            results = {"input_ids": [], "attention_mask": [], "labels": []}
            for prompt, completion in zip(example['prompt'], example['completion']):
                input_encodings = self.tokenizer.encode(f"{prompt}", truncation=True)
                target_encodings = self.tokenizer.encode(f" {completion}{self.tokenizer.eos_token}", truncation=True)
                
                input_ids = input_encodings + target_encodings
                labels = [-100] * len(input_encodings) + target_encodings
                attention_mask = [1] * len(input_ids)
                
                results["input_ids"].append(input_ids)
                results["attention_mask"].append(attention_mask)
                results["labels"].append(labels)
            return results
        else:
            # Single example processing
            input_encodings = self.tokenizer.encode(f"{example['prompt']}", truncation=True)
            target_encodings = self.tokenizer.encode(f" {example['completion']}{self.tokenizer.eos_token}", truncation=True)

            input_ids = input_encodings + target_encodings
            labels = [-100] * len(input_encodings) + target_encodings  # Mask input tokens for loss calculation
            attention_mask = [1] * len(input_ids)

            return {"input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "labels": labels}
    
    
class T5Model(MixinModel):
    """
    T5 model class
    """
    def __init__(self, checkpoint, device, peft=False, quantization=False):
        super().__init__(checkpoint, device, peft, quantization)
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, # Use 4-bit precision model loading
            bnb_4bit_quant_type="nf4", # Quantization type
            bnb_4bit_compute_dtype="float16", # Compute dtype
            bnb_4bit_use_double_quant=True, # Apply nested quantization
        ) if quantization else None

        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_path, quantization_config=bnb_config).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, clean_up_tokenization_spaces=True)
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
    def __init__(self, checkpoint, device, peft=False):
        super().__init__(checkpoint, device, peft)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, clean_up_tokenization_spaces=True)
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        self.task_type = TaskType.CAUSAL_LM
        self.target_modules = None
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
    def __init__(self, checkpoint, device, peft=False):
        super().__init__(checkpoint, device, peft)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, clean_up_tokenization_spaces=True)
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