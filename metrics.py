import evaluate
from functools import partial

"""High level implementation of each evaluation metric"""

def accuracy(references, predictions):
    acc_metric = evaluate.load("accuracy")
    return acc_metric.compute(references=references, predictions=predictions)

def sacrebleu(references, predictions):
    bleu_metric = evaluate.load("sacrebleu")
    score = bleu_metric.compute(references=references, predictions=predictions)["score"] / 100
    return {"score": score}

def squad(references, predictions):
    squad_metric = evaluate.load("squad")
    return squad_metric.compute(predictions=predictions, references=references)

def rouge(references, predictions, rouge_type):
    rouge_metric = evaluate.load('rouge')
    return rouge_metric.compute(predictions=predictions,
                  references=references,
                  rouge_types=[rouge_type],
                  use_aggregator=True)
    
rouge1 = partial(rouge, rouge_type="rouge1")
rougeLsum = partial(rouge, rouge_type="rougeLsum")
    
METRIC = {
    'anli': accuracy,
    'common_gen': rouge1,
    'squad': rougeLsum, #squad - qa metrics
    'cosmos_qa': accuracy,
    'coqa': rougeLsum, #squad - qa metrics
    'python_code': sacrebleu,
    'xsum': rougeLsum,
    'bool_q': accuracy,
    'eng_spa': sacrebleu,
    'paws': accuracy,
    'quora': rougeLsum,
    'alpaca': rougeLsum
}