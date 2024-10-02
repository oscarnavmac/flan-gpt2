from data_utils import create_instruct_dataset
from distill_utils import KLDivTrainer, DistillationArguments
from model_utils import T5Model
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = "google/flan-t5-small"
teacher_name = "google/flan-t5-base"
repo_name = "flan-t5-distill-test"

student = T5Model(checkpoint, device)
teacher = T5Model(teacher_name, device)

# Load instruct dataset (4 tasks)
datasets_names = ["common_gen", "xsum", "bool_q", "anli"]
dataset = create_instruct_dataset(datasets_names)

# Tokenize examples and initialize clm data collator
tokenized_dataset = dataset.map(
    student.tokenize_function,
    remove_columns=["prompt", "completion"],
    batched=True
)

# Define training arguments
training_args = DistillationArguments(
    output_dir=repo_name,
    learning_rate=5e-4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps = 4,
    num_train_epochs=1,
    weight_decay=0.01,
    max_steps=100, #quitar
    save_steps=1e6,
    #save_strategy='epoch',
    logging_steps=20,
    use_cpu=False, #CHECK PLEASE
    #push_to_hub=True,
    #hub_model_id=repo_name, 
    alpha=0.5,
    temperature=2.0
)

# Initialize API Trainer()
trainer = KLDivTrainer(
    student.get_model(),
    training_args,
    teacher_model=teacher.get_model(),
    data_collator=student.get_collator(),
    train_dataset=tokenized_dataset,
    tokenizer=student.get_tokenizer()
)

# Train!!!
trainer.train()