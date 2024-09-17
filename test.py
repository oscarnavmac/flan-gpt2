import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from eval_utils import Evaluation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = "gpt2-multitask-4"

model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

eval = Evaluation(model, tokenizer, device)

res_anli = eval.anli()
res_boolq = eval.bool_q()
res_commongen = eval.common_gen()
res_xsum = eval.xsum()

print(f"Total accuracy on ANLI is {res_anli}")
print(f"Total accuracy on BoolQ is {res_boolq}")
print(f"Rouge-1 score on Common Gen is {res_commongen}")
print(f"Rouge-LSum score on XSum is {res_xsum}")
