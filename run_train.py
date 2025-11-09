from models.vanilla_ft import VanillaFT
from models.uld import ULD
from models.model_utils import GPT2Model, T5Model, PythiaModel, SmolLMModel
from data.data_utils import create_instruct_dataset
import argparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()

parser.add_argument("--distill", action="store_true",
                    help="Whether to use knowledge distillation")
parser.add_argument("--lora", action="store_true",
                    help="Whether to use LoRA (Low-Rank Adaptation) training")
parser.add_argument("-m", "--model", type=str, default="gpt", const="gpt", 
                    nargs="?", choices=["gpt", "t5", "pythia", "smol"],
                    help="Foundational model to use for training")
parser.add_argument("-c", "--checkpoint", type=str, default="openai-community/gpt2-medium",
                    help="Checkpoint to use for training")
parser.add_argument("-n", "--num_samples", type=int, default=10000,
                    help="Number of samples to train on")
parser.add_argument("--num_epochs", type=int, default=1,
                    help="Number of epochs to train for")
parser.add_argument("--batch_size", type=int, default=4,
                    help="Batch size for training")
parser.add_argument("--save_model", action="store_true",
                    help="Whether to save the model")
parser.add_argument("--push_to_hub", action="store_true",
                    help="Whether to push the model to the Hub")
parser.add_argument("--repo_name", type=str, default=None,
                    help="Name of the repository to save the model to")
# LoRA-specific arguments
parser.add_argument("--lora_r", type=int, default=16,
                    help="LoRA rank (default: 16)")
parser.add_argument("--lora_alpha", type=int, default=32,
                    help="LoRA alpha parameter (default: 32)")
parser.add_argument("--lora_dropout", type=float, default=0.1,
                    help="LoRA dropout rate (default: 0.1)")
parser.add_argument("--lora_bias", type=str, default="none", choices=["none", "all", "lora_only"],
                    help="Bias handling in LoRA (default: none)")
args = parser.parse_args()


# Load instruct dataset (12 tasks) 
datasets_names = ["common_gen", "anli", "bool_q", "xsum", 
                  "python_code", "cosmos_qa", "squad", "samsum", 
                  "eng_spa", "paws"]#, "quora", "alpaca"]
dataset = create_instruct_dataset(args.num_samples, datasets_names)

# Load model
if args.model == "gpt":
    try:
        model = GPT2Model(args.checkpoint, device, peft=args.lora)
    except:
        raise ValueError("Invalid checkpoint for GPT-2 model")
elif args.model == "t5":
    try:
        model = T5Model(args.checkpoint, device, peft=args.lora)
    except:
        raise ValueError("Invalid checkpoint for T5 model")
elif args.model == "pythia":
    try:
        model = PythiaModel(args.checkpoint, device, peft=args.lora)
    except:
        raise ValueError("Invalid checkpoint for Pythia model")
elif args.model == "smol":
    try:
        model = SmolLMModel(args.checkpoint, device, peft=args.lora)
    except:
        raise ValueError("Invalid checkpoint for SmolLM model")
    
print("Model loaded with checkpoint: ", args.checkpoint)

# LoRA parameters
lora_params = {
    "rank": args.lora_r,
    "alpha": args.lora_alpha,
    "dropout": args.lora_dropout,
    "bias": args.lora_bias
}

# Repo name
if args.distill:
    postfix = "-distill"
else:
    postfix = "-ft"
repo_name = args.repo_name if args.repo_name is not None else "flan-" + str(args.checkpoint).split("/")[-1] + postfix

# Train model
if args.distill:
    checkpoint = "google/flan-t5-xl"
    teacher_model = T5Model(checkpoint, device, quantization=True)  # Quantized teacher model
    
    print("Training using Knowledge Distillation!")
    print("Teacher Model: ", checkpoint)
    uld = ULD(model, teacher_model, dataset, repo_name, device, batch_size=args.batch_size)
    uld.train(alpha=0.5, temperature=1.2, num_epochs=args.num_epochs, peft=args.lora,
              lora_params=lora_params, save_model=args.save_model, push_to_hub=args.push_to_hub)
else:
    print("Training WITHOUT distillation, vanilla fine-tuning instead!")
    vanilla_ft = VanillaFT(model, dataset, repo_name, device, batch_size=args.batch_size)
    vanilla_ft.train(num_epochs=args.num_epochs, peft=args.lora, 
                     lora_params=lora_params, save_model=args.save_model, push_to_hub=args.push_to_hub)

# Example usage:
# Vanilla fine-tuning:
# python run_train.py -n 10 --num_epochs 1 --save_model --push_to_hub
# S
# LoRA training:
# python run_train.py --lora -m gpt -c openai-community/gpt2-medium -n 1000 --num_epochs 1 --save_model --push_to_hub
# python run_train.py --lora -m gpt -c openai-community/gpt2-medium -n 1000 --lora_r 8 --lora_alpha 16 --lora_dropout 0.05
#
# Knowledge distillation:
# python run_train.py --distill -m gpt -c openai-community/gpt2-medium -n 100 --num_epochs 1
# 
# Background execution:
# nohup python run_train.py --lora -n 2000 --num_epochs 1 --save_model > lora_results.log 2>&1 &
# nohup python run_train.py -m smol -c HuggingFaceTB/SmolLM-135M -n 2000 --num_epochs 1 --save_model --push_to_hub > results.log 2>&1 &