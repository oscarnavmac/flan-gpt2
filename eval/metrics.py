import evaluate
import code_bert_score
from functools import partial
import spacy
import re

nlp = spacy.load("en_core_web_sm")
clean_code = lambda text: re.sub(r"^```[\w]*\n|```$", "", text, flags=re.MULTILINE).strip()

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

def bertscore(references, predictions, language):
    bert_metric = evaluate.load("bertscore")
    results = bert_metric.compute(
        predictions=predictions,
        references=references,
        lang=language
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

def code_bleu(references, predictions, code_lang):
    """
    Compute CodeBLEU metric for code generation tasks.
    CodeBLEU combines BLEU with syntactic information for better code evaluation.
    """
        
    code_metric = evaluate.load("k4black/codebleu")

    results = code_metric.compute(
        references=[clean_code(ref) for ref in references], 
        predictions=[clean_code(pred) for pred in predictions], 
        lang=[code_lang]
    )
    return {"score": results["codebleu"]}

def code_bertscore(references, predictions, code_lang):
    """
    Compute CodeBERTScore metric for code generation tasks.
    CodeBERTScore uses pre-trained CodeBERT embeddings to evaluate code similarity.
    """
    results = code_bert_score.score(
        cands=[clean_code(pred) for pred in predictions], 
        refs=[clean_code(ref) for ref in references], 
        lang=code_lang)
    return {"score": results[0][0].item()} # Return Precision only

rouge1 = partial(rouge, rouge_type="rouge1")
rougeLsum = partial(rouge, rouge_type="rougeLsum")
bertscore_spanish = partial(bertscore, language="es")
bertscore_english = partial(bertscore, language="en")
code_bertscore_python = partial(code_bertscore, code_lang="python")
code_bleu_python = partial(code_bleu, code_lang="python")
    
METRIC = {
    'anli': accuracy,
    'common_gen': coverage, # rougeLsum,
    'squad': rougeLsum, #squad - qa metrics
    'cosmos_qa': accuracy,
    'coqa': rougeLsum, #squad - qa metrics (REMOVE?)
    'python_code': code_bertscore_python, # code_bleu_python
    'xsum': rougeLsum, # bertscore_english,
    'bool_q': accuracy,
    'eng_spa': bertscore_spanish, # sacrebleu,
    'paws': accuracy,
    'quora': rougeLsum,
    'alpaca': rougeLsum
}