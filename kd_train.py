from tqdm.auto import tqdm
from transformers import get_linear_schedule_with_warmup
from data_utils import create_instruct_dataset
from model_utils import T5Model, GPT2Model
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_ 
from torch.optim import AdamW
import torch
import logging
import pickle
from torch.amp import autocast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = 'openai-community/gpt2-medium'
teacher_name = 'google/flan-t5-large'
repo_name = "flan-gpt2-medium-distill_V2"

student = GPT2Model(checkpoint, device)
teacher = T5Model(teacher_name, device)

# Get models
student_model = student.get_model()
teacher_model = teacher.get_model()

student_model.to(torch.bfloat16)
student_model.gradient_checkpointing_enable()

# Load instruct dataset (4 tasks)
datasets_names = ["common_gen", "anli", "bool_q", "xsum", "python_code", "cosmos_qa", "squad", "coqa", "eng_spa", "paws"]
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

tokenized_teacher_dataset.set_format("torch")
tokenized_student_dataset.set_format("torch")

# Get Training DataLoader
batch_size = 1 # For now we can only use stochastic training
train_teacher_dataloader = DataLoader(
    tokenized_teacher_dataset, shuffle=False, batch_size=batch_size, collate_fn=teacher.get_collator()
)
train_student_dataloader = DataLoader(
    tokenized_student_dataset, shuffle=False, batch_size=batch_size, collate_fn=student.get_collator()
)

# Define hyperparameters:
optimizer = AdamW(student_model.parameters(), lr=5e-4, weight_decay=0.01)

push_to_hub = False
save_model = True
logging_steps = 100
save_steps = 100000
gradient_accumulation_steps = 4

# Distillation hyperparameters
alpha = 0.80
temperature = 1.0

num_epochs = 1
num_training_steps = num_epochs * len(train_student_dataloader) // gradient_accumulation_steps
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)

progress_bar = tqdm(range(num_training_steps))
logging.basicConfig(level=logging.INFO)

# TRAIN!!!!!
losses = []
teacher_model.eval()
student_model.train()
step = 0
global_step = 0
running_loss = 0
for epoch in range(num_epochs):
    for student_batch, teacher_batch in zip(train_student_dataloader, train_teacher_dataloader):
        #print("STUDENT")
        #print({k: (v.shape,v) for k, v in student_batch.items()})
        #print("TEACHER")
        #print({k: (v.shape,v) for k, v in teacher_batch.items()})
        
        student_batch = {k: v.to(device) for k, v in student_batch.items()}
        teacher_batch = {k: v.to(device) for k, v in teacher_batch.items()}
        
        student_targets = student_batch["targets"]
        student_batch.pop("targets")
        
        #with autocast("cuda", dtype=torch.bfloat16):
        student_outputs = student_model(**student_batch)
        student_loss = student_outputs.loss
        
        with torch.no_grad():
            teacher_outputs = teacher_model(**teacher_batch)
            
        student_logits = student_outputs.logits
        teacher_logits = teacher_outputs.logits
        
        #print("LOGITS")
        #print(student_logits.size())
        #print(student_logits)
        #print(teacher_logits.size())
        #print(teacher_logits)
        
        # Apply Softmax to get each probabily distribution
        student_probs = F.softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        
        #max_length = max(max(student_answer_size), max(teacher_answer_size))
        #print(max_length)
        
        # Warning: this is hardcoded!!!
        target_idx = student_batch["input_ids"].size(1) - student_targets.size(1) - 1 # Consider EOS token too
        
        #is_value = student_batch["labels"].eq(-100)
        #target_idx = int(is_value.sum())
        student_probs = student_probs[:, target_idx:, :]
        #print(target_idx)
        #print(student_probs)
        #break
        
        #print()
        #print(tokeni.decode(student_batch["input_ids"][0]))
        #print("ANSWER $$$$$$$$")
        #print(tokeni.decode(student_targets[0])) 
        #print("OK-------------------------------------------------")
        #print(tokeni2.decode(teacher_batch["input_ids"][0]))
        #print("ANSWER $$$$$$$$")
        #print(tokeni2.decode(teacher_batch["labels"][0]))
        
        #print("PROBS")
        #print(student_probs.size())
        #print(student_probs)
        #print(teacher_probs.size())
        #print(teacher_probs)

        
        # Sort in descending order to align probabilities
        student_probs = student_probs.sort(dim=-1, descending=True).values
        teacher_probs = teacher_probs.sort(dim=-1, descending=True).values
        
        # Pad to get same vocabulary size
        diff_size = student_probs.size(2) - teacher_probs.size(2)
        if diff_size > 0:
            teacher_probs = F.pad(teacher_probs, (0, diff_size), value=0)
        elif diff_size < 0:
            student_probs = F.pad(student_probs, (0, abs(diff_size)), value=0)
            
        #print("NEW PROBS AFTER PADDING")
        #print(student_probs.size())
        #print(student_probs)
        #print(teacher_probs.size())
        #print(teacher_probs)
        
        #break
        
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
        
        loss /= gradient_accumulation_steps
        
        
        # Vanilla training from here
        
        running_loss += loss.item() #/ gradient_accumulation_steps
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            if (global_step + 1) % logging_steps == 0:
                loss = running_loss / logging_steps
                losses.append(float(loss))
                running_loss = 0
                
                total_norm = student.get_global_grad_norm()
                
                logging.info(f"Loss: {loss}, lr: {scheduler.get_lr()}, grad_norm: {total_norm}, step: {global_step}")
                print(f"Student Loss {student_loss.item() / gradient_accumulation_steps}")

            clip_grad_norm_(student_model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            global_step+=1
        step+=1

        if save_model and (global_step + 1) % save_steps == 0:
            logging.info("saving model...")
            student_model.save_pretrained(repo_name)
            student.get_tokenizer().save_pretrained(repo_name)

if save_model:
    student_model.save_pretrained(repo_name)
    student.get_tokenizer().save_pretrained(repo_name)

# Push the model to the repo
if push_to_hub:
    student_model.push_to_hub(repo_name, commit_message=f"Uploaded/updated model with loss {losses[-1]}")
    student.get_tokenizer().push_to_hub(repo_name)
    
# Saving model losses
with open('losses_kd.pkl', 'wb') as f:
    pickle.dump(losses, f)