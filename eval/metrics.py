import evaluate
from functools import partial
import spacy

nlp = spacy.load("en_core_web_sm")

"""High level implementation of each evaluation metric"""

def accuracy(references, predictions):
    """Compute accuracy metric"""
    acc_metric = evaluate.load("accuracy")
    return acc_metric.compute(references=references, predictions=predictions)

def sacrebleu(references, predictions):
    """Compute SacreBLEU metric"""
    bleu_metric = evaluate.load("sacrebleu")
    score = bleu_metric.compute(references=references, predictions=predictions)["score"] / 100
    return {"score": score}

def bertscore(references, predictions):
    bert_metric = evaluate.load("bertscore")
    results = bert_metric.compute(
        predictions=predictions,
        references=references,
        lang="es"
    )
    return {"score": sum(results["f1"]) / len(results["f1"])}


def squad(references, predictions):
    squad_metric = evaluate.load("squad")
    return squad_metric.compute(predictions=predictions, references=references)

def rouge(references, predictions, rouge_type):
    rouge_metric = evaluate.load('rouge')
    return rouge_metric.compute(predictions=predictions,
                  references=references,
                  rouge_types=[rouge_type],
                  use_aggregator=True)
    
def coverage(concepts, predictions):
    """Average fraction of input concepts appearing (by lemma) in each prediction."""
    scores = []
    for concept_list, pred in zip(concepts, predictions):
        pred_lemmas = {token.lemma_.lower() for token in nlp(pred)}
        covered = sum(1 for c in concept_list if c.lower() in pred_lemmas)
        scores.append(covered / len(concept_list))
    return {"score": sum(scores) / len(scores)}
    
rouge1 = partial(rouge, rouge_type="rouge1")
rougeLsum = partial(rouge, rouge_type="rougeLsum")
    
METRIC = {
    'anli': accuracy,
    'common_gen': coverage,
    'squad': rougeLsum, #squad - qa metrics
    'cosmos_qa': accuracy,
    'coqa': rougeLsum, #squad - qa metrics
    'python_code': sacrebleu,
    'xsum': rougeLsum,
    'bool_q': accuracy,
    'eng_spa': bertscore, #sacrebleu,
    'paws': accuracy,
    'quora': rougeLsum,
    'alpaca': rougeLsum
}