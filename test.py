import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_utils import Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = "gpt2-multitask-4"

model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

eval = Evaluation(model, tokenizer, device)

eval.anli()
eval.bool_q()
