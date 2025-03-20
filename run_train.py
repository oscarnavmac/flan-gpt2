from models.vanilla_ft import VanillaFT
from models.model_utils import GPT2Model, T5Model, PythiaModel
from data.data_utils import create_instruct_dataset
import argparse
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()

parser.add_argument("--distill", action="store_true",
                    help="Whether to use knowledge distillation")
parser.add_argument("-m", "--model", type=str, default="gpt", const="gpt", 
                    nargs="?", choices=["gpt", "t5", "pythia"],
                    help="Foundational model to use for training")
parser.add_argument("-c", "--checkpoint", type=str, default="openai-community/gpt2-medium",
                    help="Checkpoint to use for training")
parser.add_argument("-n", "--num_samples", type=int, default=10000,
                    help="Number of samples to train on")
parser.add_argument("--num_epochs", type=int, default=1,
                    help="Number of epochs to train for")
parser.add_argument("--save_model", action="store_true",
                    help="Whether to save the model")
parser.add_argument("--push_to_hub", action="store_true",
                    help="Whether to push the model to the Hub")
parser.add_argument("--repo_name", type=str, default=None,
                    help="Name of the repository to save the model to")
args = parser.parse_args()


# Load instruct dataset (4 tasks) 
datasets_names = ["common_gen", "anli", "bool_q", "xsum", 
                  "python_code", "cosmos_qa", "squad", "coqa", 
                  "eng_spa", "paws"]
dataset = create_instruct_dataset(args.num_samples, datasets_names)

# Load model
if args.model == "gpt":
    try:
        model = GPT2Model(args.checkpoint, device)
        print("Model loaded with checkpoint: ", args.checkpoint)
    except:
        raise ValueError("Invalid checkpoint for GPT-2 model")
elif args.model == "t5":
    try:
        model = T5Model(args.checkpoint, device)
    except:
        raise ValueError("Invalid checkpoint for T5 model")
else:
    try:
        model = PythiaModel(args.checkpoint, device)
    except:
        raise ValueError("Invalid checkpoint for Pythia model")

# Repo name
postfix = "-distill" if args.distill else "-ft"
repo_name = args.repo_name if args.repo_name is not None else "flan-" + str(args.checkpoint).split("/")[-1] + postfix

# Train model
if args.distill:
    print("Training using Knowledge Distillation!")
    
else:
    print("Training WITHOUT distillation, vanilla fine-tuning instead!")
    vanilla_ft = VanillaFT(model, dataset, repo_name, device)
    vanilla_ft.train(num_epochs=args.num_epochs, save_model=args.save_model, push_to_hub=args.push_to_hub)

# Example usage:
# python run_train.py --distill -m gpt -c openai-community/gpt2-medium -n 100 --num_epochs 1
# python run_train.py -n 10 --num_epochs 1 --save_model --push_to_hub