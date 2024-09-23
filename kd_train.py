from data_utils import create_instruct_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import TrainingArguments, Trainer
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = 'openai-community/gpt2-medium'
teacher_name = '"google/flan-t5-large"'
repo_name = "gpt2-multitask-4"


# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
tokenizer.pad_token = tokenizer.eos_token

# Load instruct dataset (4 tasks)
datasets_names = ["common_gen", "xsum", "bool_q", "anli"]
dataset = create_instruct_dataset(datasets_names)

# Tokenize examples and initialize clm data collator
tokenized_dataset = dataset.map(
    lambda example:  tokenizer(example["text"], truncation=True),
    remove_columns=["text"],
    batched=True
)
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir=repo_name,
    learning_rate=5e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps = 4,
    num_train_epochs=3,
    weight_decay=0.01,
    #max_steps=10, #quitar
    save_strategy='epoch',
    logging_steps=50,
    use_cpu=False,
    push_to_hub=True, #CHECK PLEASE
    hub_model_id=repo_name, 
)

# Initialize API Trainer()
trainer = Trainer(
    model,
    training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# Train!!!
trainer.train()