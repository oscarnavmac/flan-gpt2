from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.utils.data import DataLoader
import torch
import os
import logging
import pickle
from torch.amp import autocast
from peft import LoraConfig, get_peft_model, TaskType
from models.model_utils import MixinModel

class VanillaFT:
    def __init__(self, wrapped_model, dataset, repo_name, device, repo_dir="output", batch_size=1):
        self.wrapped_model: MixinModel = wrapped_model
        self.dataset = dataset
        self.repo_name = repo_name
        self.device = device
        self.train_dataloader = self.preprocess(batch_size)
        self.repo_dir = repo_dir
        
    def preprocess(self, batch_size):
        # Tokenize examples
        tokenized_dataset = self.dataset.map(
            self.wrapped_model.tokenize_function,
            remove_columns=["prompt", "completion"],
            batched=True,
            batch_size=500
        )
        tokenized_dataset.set_format("torch")

        # Get Training DataLoader
        return DataLoader(
            tokenized_dataset, shuffle=False, batch_size=batch_size, collate_fn=self.wrapped_model.get_collator()
        )
        
    def save_model(self, model, path):
        logging.info(f"saving model to {path} ...")
        model.save_pretrained(path)
        self.wrapped_model.get_tokenizer().save_pretrained(path)

    def train(self, num_epochs=1, peft=False, lora_params=None, save_model=True, push_to_hub=True,
            logging_steps=100, save_steps=1000000, gradient_accumulation_steps=4, max_steps=None):

        model = self.wrapped_model.get_model(peft=peft, lora_params=lora_params)
        if peft:
           logging.info(f"Training with LoRA parameters: {lora_params}")
        self.wrapped_model.print_trainable_parameters()
        model.gradient_checkpointing_enable()
        model.to(torch.bfloat16)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() 
                            if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in model.named_parameters() 
                            if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        # Define hyperparameters:
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)

        num_training_steps = num_epochs * len(self.train_dataloader) // gradient_accumulation_steps if max_steps==None else max_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        progress_bar = tqdm(range(num_training_steps))
        logging.basicConfig(level=logging.INFO)
        repo_path = os.path.join(self.repo_dir, self.repo_name)
        os.makedirs(repo_path, exist_ok=True)
        
        logging.info(f"lr: {scheduler.get_lr()}, gradient_accumulation_steps: {gradient_accumulation_steps}")
        logging.info(f"Training {self.repo_name} for {num_epochs} epochs with {num_training_steps} steps")
        if torch.cuda.is_available():
            logging.info(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1e9} GB")

        # TRAIN!!!!!
        losses = []
        model.train()
        model.zero_grad()
        step = 0
        global_step = 0
        running_loss = 0
        for epoch in range(num_epochs):
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if "targets" in batch:
                    batch.pop("targets")
                #with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                running_loss += loss.item()
                loss.backward()
                
                #if global_step >= MAX_STEPS*gradient_accumulation_steps:
                    #break
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    
                    if (global_step + 1) % logging_steps == 0:
                        loss = running_loss / logging_steps
                        losses.append(float(loss))
                        running_loss = 0
                        
                        total_norm = self.wrapped_model.get_global_grad_norm()
                        
                        logging.info(f"Loss: {loss}, lr: {scheduler.get_lr()}, grad_norm: {total_norm}, step: {global_step}")
                        if torch.cuda.is_available():
                            logging.info(f"GPU Memory Usage: {torch.cuda.memory_allocated() / 1e9} GB")

                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                    progress_bar.update(1)
                    global_step+=1
                step+=1

                if save_model and (global_step + 1) % save_steps == 0:
                    self.save_model(model, repo_path)

        if save_model:
            self.save_model(model, repo_path)

        # Push the model to the repo
        if push_to_hub:
            model.push_to_hub(self.repo_name, commit_message=f"Uploaded/updated model with loss {losses[-1]}")
            self.wrapped_model.get_tokenizer().push_to_hub(self.repo_name)
            
        # Saving model losses
        losses_path = os.path.join(repo_path, "losses.pkl") 
        with open(losses_path, 'wb') as f:
            pickle.dump(losses, f)
            
    def lora_train(self, num_epochs=1, save_model=True, push_to_hub=True,
            logging_steps=100, save_steps=1000000, gradient_accumulation_steps=4, max_steps=None,
            lora_r=16, lora_alpha=32, lora_dropout=0.1, target_modules=None):
        """
        Train the model using LoRA (Low-Rank Adaptation)
        
        Args:
            num_epochs: Number of training epochs
            save_model: Whether to save the model
            push_to_hub: Whether to push to Hugging Face Hub
            logging_steps: Steps between logging
            save_steps: Steps between model saves
            gradient_accumulation_steps: Number of gradient accumulation steps
            max_steps: Maximum number of training steps (overrides epochs if set)
            lora_r: LoRA rank
            lora_alpha: LoRA alpha parameter
            lora_dropout: LoRA dropout rate
            target_modules: Target modules for LoRA (None for auto-detection)
        """
        model = self.wrapped_model.get_model()
        
        # Configure LoRA
        if target_modules is None:
            # Auto-detect target modules based on model type
            if hasattr(model, 'transformer'):  # GPT-2 style
                target_modules = ["c_attn", "c_proj", "c_fc"]
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):  # LLaMA/Pythia style
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
            else:
                # Default to common attention modules
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        
        # Apply LoRA to the model
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Enable gradient checkpointing and set dtype
        model.gradient_checkpointing_enable()
        model.to(torch.bfloat16)
        
        # Define optimizer (only train LoRA parameters)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

        num_training_steps = num_epochs * len(self.train_dataloader) // gradient_accumulation_steps if max_steps==None else max_steps

        scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )

        progress_bar = tqdm(range(num_training_steps))
        logging.basicConfig(level=logging.INFO)
        repo_path = os.path.join(self.repo_dir, self.repo_name)
        os.makedirs(repo_path, exist_ok=True)
        
        logging.info(f"LoRA Training Configuration:")
        logging.info(f"  - LoRA rank (r): {lora_r}")
        logging.info(f"  - LoRA alpha: {lora_alpha}")
        logging.info(f"  - LoRA dropout: {lora_dropout}")
        logging.info(f"  - Target modules: {target_modules}")
        logging.info(f"  - Learning rate: 1e-4")
        logging.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
        logging.info(f"Training {self.repo_name} for {num_epochs} epochs with {num_training_steps} steps")

        # TRAIN!!!!!
        losses = []
        model.train()
        model.zero_grad()
        step = 0
        global_step = 0
        running_loss = 0
        for epoch in range(num_epochs):
            for batch in self.train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if "targets" in batch:
                    batch.pop("targets")
                
                outputs = model(**batch)
                loss = outputs.loss / gradient_accumulation_steps
                running_loss += loss.item()
                loss.backward()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    
                    if (global_step + 1) % logging_steps == 0:
                        loss = running_loss / logging_steps
                        losses.append(float(loss))
                        running_loss = 0
                        
                        # Calculate gradient norm for LoRA parameters
                        total_norm = 0.0
                        for p in model.parameters():
                            if p.grad is not None:
                                param_norm = p.grad.detach().data.norm(2)
                                total_norm += param_norm.item() ** 2
                        total_norm = total_norm ** 0.5
                        
                        logging.info(f"Loss: {loss}, lr: {scheduler.get_lr()}, grad_norm: {total_norm}, step: {global_step}")
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                    progress_bar.update(1)
                    global_step+=1
                step+=1

                if save_model and (global_step + 1) % save_steps == 0:
                    # Save LoRA adapter
                    logging.info(f"saving LoRA adapter to {repo_path} ...")
                    model.save_pretrained(repo_path)
                    self.wrapped_model.get_tokenizer().save_pretrained(repo_path)

        if save_model:
            # Save LoRA adapter
            logging.info(f"saving final LoRA adapter to {repo_path} ...")
            model.save_pretrained(repo_path)
            self.wrapped_model.get_tokenizer().save_pretrained(repo_path)

        # Push the adapter to the repo
        if push_to_hub:
            model.push_to_hub(self.repo_name, commit_message=f"Uploaded/updated LoRA adapter with loss {losses[-1]}")
            self.wrapped_model.get_tokenizer().push_to_hub(self.repo_name)
            
        # Saving model losses
        losses_path = os.path.join(repo_path, "losses_lora.pkl") 
        with open(losses_path, 'wb') as f:
            pickle.dump(losses, f)
            
        return model
    
    @staticmethod
    def load_lora_model(base_model_path, lora_adapter_path, device):
        """
        Load a base model with LoRA adapter for inference
        
        Args:
            base_model_path: Path to the base model
            lora_adapter_path: Path to the LoRA adapter
            device: Device to load the model on
            
        Returns:
            Model with LoRA adapter loaded
        """
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(base_model, lora_adapter_path)
        model.to(device)
        
        return model