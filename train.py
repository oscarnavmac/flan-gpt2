from tqdm.auto import tqdm
from transformers import get_scheduler, AdamW
from data_utils import create_instruct_dataset
from model_utils import GPT2Model
from torch.utils.data import DataLoader
import torch
import logging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
checkpoint = 'openai-community/gpt2'

repo_name = "gpt2-dummy-testing"

gpt2_model = GPT2Model(checkpoint, device)

# Load model and tokenizer
model = gpt2_model.get_model()

# Load instruct dataset (4 tasks)
datasets_names = ["common_gen", "xsum", "bool_q", "anli"]
dataset = create_instruct_dataset(datasets_names)

# Tokenize examples
tokenized_dataset = dataset.map(
    gpt2_model.tokenize_function,
    remove_columns=["prompt", "completion"],
    batched=False
)

# Get Training DataLoader
tokenized_dataset.set_format("torch")

train_dataloader = DataLoader(
    tokenized_dataset, shuffle=True, batch_size=4, collate_fn=gpt2_model.get_collator()
)

# Define hyperparameters:
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 2
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
push_to_hub = True
save_model = True
logging_steps = 100
save_steps = 1000

progress_bar = tqdm(range(num_training_steps))
logging.basicConfig(level=logging.INFO)

# TRAIN!!!!!
model.train()
global_step = 0
for epoch in range(num_epochs):
    for batch in train_dataloader:
        #print({k: (v.shape,v) for k, v in batch.items()})
        #break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
        global_step+=1

        if global_step % logging_steps == 0:
            logging.info(f"Loss: {loss}, step: {global_step}")

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