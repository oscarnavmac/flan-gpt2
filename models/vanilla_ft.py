from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, get_polynomial_decay_schedule_with_warmup
from torch.utils.data import DataLoader
import torch
import os
import logging
import pickle
from torch.amp import autocast

class VanillaFT:
    def __init__(self, wrapped_model, dataset, repo_name, device, repo_dir="output", batch_size=1):
        self.wrapped_model = wrapped_model
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

    def train(self, num_epochs=1, save_model=True, push_to_hub=True,
            logging_steps=100, save_steps=1000000, gradient_accumulation_steps=4, max_steps=None):
        
        model = self.wrapped_model.get_model()
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
                batch.pop("targets") # Ignore this, is for other purposes which is not important right now
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