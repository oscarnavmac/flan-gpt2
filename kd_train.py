from tqdm.auto import tqdm
from transformers import get_scheduler
from data_utils import create_instruct_dataset
from model_utils import T5Model, GPT2Model
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.optim import AdamW
import torch
import logging
import pickle

#TODO: Implement Batch training

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = 'openai-community/gpt2'
teacher_name = "google/flan-t5-base"
repo_name = "flan-t5-distill-test"

student = GPT2Model(checkpoint, device)
teacher = T5Model(teacher_name, device)

# Get models
student_model = student.get_model()
teacher_model = teacher.get_model()

# Load instruct dataset (4 tasks)
datasets_names = ["common_gen", "xsum", "bool_q", "anli"]
dataset = create_instruct_dataset(datasets_names)

# Tokenize datasets
tokenized_teacher_dataset = dataset.map(
    teacher.tokenize_function,
    remove_columns=["prompt", "completion"],
    desc="Tokenizing teacher inputs",
    batched=True,
    batch_size=1000
)
tokenized_student_dataset = dataset.map(
    student.tokenize_function,
    remove_columns=["prompt", "completion"],
    desc="Tokenizing student inputs",
    batched=True,
    batch_size=500
)

#tokeni = student.get_tokenizer()
#tokeni2 = teacher.get_tokenizer()
#print(tokeni.decode(tokenized_student_dataset["input_ids"][0]))

#tokenized_teacher_dataset.set_format("torch")
#tokenized_student_dataset.set_format("torch")

# Get Training DataLoader
batch_size = 1 # For now we can only use stochastic training
train_teacher_dataloader = DataLoader(
    tokenized_teacher_dataset, shuffle=False, batch_size=batch_size, collate_fn=teacher.get_collator()
)
train_student_dataloader = DataLoader(
    tokenized_student_dataset, shuffle=False, batch_size=batch_size, collate_fn=student.get_collator()
)
train_dataloader = zip(train_student_dataloader, train_teacher_dataloader)

# Define hyperparameters:
alpha = 0.5
temperature = 1.0
optimizer = AdamW(student_model.parameters(), lr=5e-4)
num_epochs = 2
num_training_steps = num_epochs * len(train_student_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
push_to_hub = False
save_model = False
logging_steps = 100
save_steps = 1000

progress_bar = tqdm(range(num_training_steps))
logging.basicConfig(level=logging.INFO)

# TRAIN!!!!!
losses = []
teacher_model.eval()
student_model.train()
global_step = 0
for epoch in range(num_epochs):
    for student_batch, teacher_batch in train_dataloader:
        #print("STUDENT")
        #print({k: (v.shape,v) for k, v in student_batch.items()})
        #print("TEACHER")
        #print({k: (v.shape,v) for k, v in teacher_batch.items()})
        
        student_batch = {k: v.to(device) for k, v in student_batch.items()}
        teacher_batch = {k: v.to(device) for k, v in teacher_batch.items()}
        
        student_targets = student_batch["targets"]
        student_batch.pop("targets")
        
        student_outputs = student_model(**student_batch)
        student_loss = student_outputs.loss
        
        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_batch)
            
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        
        #THEY NOT REALLY LOGITS ANYMORE SO PLEASE CHANGE THE VARIABLES NAMES 
        student_probs = F.softmax(student_logits, dim=-1)
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        
        
        #print("So Far So Good")
        #print(student_logits.size())
        #print(student_logits)
        #print(teacher_logits.size())
        #print(teacher_logits)
        #break
        
        #max_length = max(max(student_answer_size), max(teacher_answer_size))
        #print(max_length)
        
        # Warning: this is hardcoded!!!
        target_idx = student_batch["input_ids"].size(1) - student_targets.size(1)
        student_probs = student_probs[:, target_idx:, :]
        #print(target_idx)
        
        #print()
        #print(tokeni.decode(student_batch["input_ids"][0]))
        #print("ANSWER $$$$$$$$")
        #print(tokeni.decode(student_targets[0])) 
        #print("OK-------------------------------------------------")
        #print(tokeni2.decode(teacher_batch["input_ids"][0]))
        #print("ANSWER $$$$$$$$")
        #print(tokeni2.decode(teacher_batch["labels"][0]))
        
        # Sort in descending order to align probabilities
        student_probs = student_probs.sort(dim=-1, descending=True).values
        teacher_probs = teacher_probs.sort(dim=-1, descending=True).values
        
        # Pad to get same vocabulary size
        diff_size = student_probs.size(2) - teacher_probs.size(2)
        if diff_size > 0:
            teacher_probs = F.pad(teacher_probs, (0, diff_size), value=0)
        elif diff_size < 0:
            student_probs = F.pad(student_probs, (0, abs(diff_size)), value=0)
            
        #print("WE ARE ALMOST THERE")
        #print(student_logits.size())
        #print(student_logits)
        #print(teacher_logits.size())
        #print(teacher_logits)
        
        #print("CALCULATING LOSS")
        
        distillation_loss = torch.zeros(student_probs.size(0), device=student_model.device)
        for i in range(student_probs.size(0)):
            size = min(student_probs.size(1), teacher_probs.size(1))
            #print(size)
            #print(student_logits[i][:size])
            #print(teacher_logits[i][:size])
            #print(f"size: {abs(student_logits[i][:size] - teacher_logits[i][:size]).size()}")
            #print(f"size: {abs(student_logits[i][:size] - teacher_logits[i][:size]).sum(-1).size()}")
            #print(f"size: {abs(student_logits[i][:size] - teacher_logits[i][:size]).sum(-1).mean(-1).size()}")
            distillation_loss[i] = abs(student_probs[i][:size] - teacher_probs[i][:size]).sum(-1).mean(-1)
        distillation_loss = distillation_loss.mean()
        
        loss = alpha * student_loss + (1-alpha) * distillation_loss
        
        #print(distillation_loss)
        #print(student_loss)
        #print(float(loss))
        
        #break
    
        losses.append(loss)
        
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
            student_model.save_pretrained(repo_name)
            student.get_tokenizer().save_pretrained(repo_name)
            
# Saving model losses
with open('losses.pkl', 'wb') as f:
    pickle.dump(losses, f)

if save_model:
    student_model.save_pretrained(repo_name)
    student.get_tokenizer().save_pretrained(repo_name)

# Push the model to the repo
if push_to_hub:
    student_model.push_to_hub(repo_name)
    student.get_tokenizer().push_to_hub(repo_name)