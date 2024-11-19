import evaluate
from functools import partial

"""High level implementation of each evaluation metric"""

def accuracy(references, predictions):
    acc_metric = evaluate.load("accuracy")
    return acc_metric.compute(references=references, predictions=predictions)

def bleu(references, predictions):
    pass

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
    'squad': None, #squad - qa metrics
    'cosmos_qa': accuracy,
    'coqa': None, #squad - qa metrics
    'human_eval': None,
    'xsum': rougeLsum,
    'bool_q': accuracy
}