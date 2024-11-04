from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup, AdamW, get_polynomial_decay_schedule_with_warmup
from data_utils import create_instruct_dataset
from model_utils import GPT2Model
from torch.utils.data import DataLoader
import torch
import logging
import pickle
from torch.amp import autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
checkpoint = 'openai-community/gpt2-medium'

repo_name = "gpt2-dummy-testing"

gpt2_model = GPT2Model(checkpoint, device)

# Load model and tokenizer
model = gpt2_model.get_model()
model.gradient_checkpointing_enable()
#model.to(torch.bfloat16)

tokenizer = gpt2_model.get_tokenizer()

# Load instruct dataset (4 tasks)
datasets_names = ["common_gen"]#, "xsum", "bool_q", "anli"]
dataset = create_instruct_dataset(datasets_names)

# Tokenize examples
tokenized_dataset = dataset.map(
    gpt2_model.tokenize_function,
    remove_columns=["prompt", "completion"],
    batched=True,
    batch_size=500
)

# Get Training DataLoader
tokenized_dataset.set_format("torch")

train_dataloader = DataLoader(
    tokenized_dataset, shuffle=False, batch_size=1, collate_fn=gpt2_model.get_collator()
)

# Define hyperparameters:
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.01)


push_to_hub = False
save_model = False
logging_steps = 25
save_steps = 1000
gradient_accumulation_steps = 4

#logging_steps *= gradient_accumulation_steps

MAX_STEPS = 250

num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader) // gradient_accumulation_steps

# Overwritting num_training_steps
if MAX_STEPS is not None:
    num_training_steps = MAX_STEPS

scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))
logging.basicConfig(level=logging.INFO)

# TRAIN!!!!!
losses = []
model.train()
global_step = 1
step = 1
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch.pop("targets") # Ignore this, is for other purposes which is not important right now
        #with autocast("cuda", dtype=torch.bfloat16):
        outputs = model(**batch)
        loss = outputs.loss #/ gradient_accumulation_steps
        loss.backward()
        
        step+=1
        
        if global_step >= MAX_STEPS:
            break
        
        if step % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            #print(loss) 
            progress_bar.update(1)
            global_step+=1

            if global_step % logging_steps == 0:
                losses.append(float(loss))
                
                total_norm = 0.0
                parameters = [p for p in model.parameters() if p.grad is not None and p.requires_grad]
                for p in parameters:
                    print("esto nunca ocurre")
                    param_norm = p.grad.detach().data.norm(2)
                    total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                logging.info(f"Loss: {loss}, lr: {scheduler.get_lr()}, grad_norm: {total_norm}, step: {global_step}")

            if save_model and global_step % save_steps == 0:
                logging.info("saving model...")
                model.save_pretrained(repo_name)
                gpt2_model.get_tokenizer().save_pretrained(repo_name)
                

if save_model:
    model.save_pretrained(repo_name)
    gpt2_model.get_tokenizer().save_pretrained(repo_name)

# Push the model to the repo
if push_to_hub:
    model.push_to_hub(repo_name)
    gpt2_model.get_tokenizer().push_to_hub(repo_name)
    
# Saving model losses
with open('losses_ft.pkl', 'wb') as f:
    pickle.dump(losses, f)