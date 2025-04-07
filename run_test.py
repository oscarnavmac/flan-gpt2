from eval.eval_utils import Evaluation
from models.model_utils import GPT2Model, T5Model, PythiaModel, SmolLMModel
from data.data_utils import create_instruct_dataset
import argparse
import csv
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="gpt", const="gpt", 
                    nargs="?", choices=["gpt", "t5", "pythia", "smol"],
                    help="Model architecture to use for evaluation")
parser.add_argument("-c", "--checkpoint", type=str, default="openai-community/gpt2-medium",
                    help="Checkpoint to use for evaluation")
parser.add_argument("-n", "--num_samples", type=int, default=10000,
                    help="Number of samples to evaluate on")
parser.add_argument("--n_shot", type=int, default=0,
                    help="Number of examples to use for few-shot evaluation")
parser.add_argument("--no_save_results", action="store_true",
                    help="Whether to save the evaluation results")
parser.add_argument("--save_dir", type=str, default=".",
                    help="Directory to save the evaluation results")
args = parser.parse_args()

# Load instruct dataset (10 tasks)
datasets_names = ["common_gen", "anli", "bool_q", "xsum", 
                  "python_code", "cosmos_qa", "squad", "coqa", 
                  "eng_spa", "paws"]

# Load model
if args.model == "gpt":
    try:
        model = GPT2Model(args.checkpoint, device)
        return_full_text = True
    except:
        raise ValueError("Invalid checkpoint for GPT-2 model")
elif args.model == "t5":
    try:
        model = T5Model(args.checkpoint, device)
        return_full_text = False
    except:
        raise ValueError("Invalid checkpoint for T5 model")
elif args.model == "pythia":
    try:
        model = PythiaModel(args.checkpoint, device)
        return_full_text = True
    except:
        raise ValueError("Invalid checkpoint for Pythia model")
elif args.model == "smol":
    try:
        model = SmolLMModel(args.checkpoint, device)
        return_full_text = True
    except:
        raise ValueError("Invalid checkpoint for SmolLM model")
    
print("Model loaded with checkpoint: ", args.checkpoint)


# Load evaluation class
eval = Evaluation(model.get_model(), model.get_tokenizer(), device)

# Evaluate each task
for dataset_name in datasets_names:
    print(f"Evaluating {dataset_name}...")
    # Load dataset
    dataset = create_instruct_dataset(args.num_samples, [dataset_name])
    
    # Evaluate model
    if args.n_shot > 0:
        #dataset = dataset.select(range(args.n_shot))
        pass
        
    try:
         result = eval.evaluate(dataset_name, args.num_samples, 
                                training_set=True, return_full_text=return_full_text)
    except Exception as e:
        print(f"Error evaluating {dataset_name}: {e}")
        continue
    
    print(f"Results for {dataset_name}: {result}")
    # Save results
    if not args.no_save_results:
        # append results to CSV file
        header = ["dataset", "score"]
        with open(f"{args.save_dir}/{args.checkpoint}_results.csv", mode="a", newline="") as f:
            writer = csv.writer(f)
            if f.tell() == 0:
                writer.writerow(header)
            writer.writerow([dataset_name, result])
        print(f"Results saved to {args.save_dir}/eval_results.csv")
                
# Example usage:
# nohup python run_test.py -m smol -c OscarNav/flan-SmolLM-135M-distill -n 10 --n_shot 0 > eval_results.log 2>&1 &