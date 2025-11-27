import evaluate
import code_bert_score
from functools import partial, wraps
import spacy
import re

nlp = spacy.load("en_core_web_sm")
clean_code = lambda text: re.sub(r"^```[\w]*\n|```$", "", text, flags=re.MULTILINE).strip()

"""High level implementation of each evaluation metric"""

def accuracy(references, predictions):
    """Compute accuracy metric for classification tasks."""
    acc_metric = evaluate.load("accuracy")
    results = acc_metric.compute(references=references, predictions=predictions)
    return {"score": results["accuracy"]}

def sacrebleu(references, predictions):
    """Compute SacreBLEU metric for translation tasks."""
    bleu_metric = evaluate.load("sacrebleu")
    results = bleu_metric.compute(references=references, predictions=predictions)
    return {"score": results["score"] / 100}

def bertscore(references, predictions, language):
    """Compute BERTScore metric for text generation tasks."""
    bert_metric = evaluate.load("bertscore")
    results = bert_metric.compute(
        predictions=predictions,
        references=references,
        lang=language
    )
    return {"score": sum(results["f1"]) / len(results["f1"])}

def squad(references, predictions):
    """Compute SQuAD metric for question answering tasks."""
    squad_metric = evaluate.load("squad")
    results = squad_metric.compute(predictions=predictions, references=references)
    return {"score": results["f1"] / 100}

def rougeL(references, predictions):
    """Compute ROUGE metric for text summarization tasks."""
    rouge_metric = evaluate.load('rouge')
    results = rouge_metric.compute(predictions=predictions,
                  references=references,
                  rouge_types=["rougeL"],
                  use_aggregator=True)
    return {"score": results["rougeL"]}

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

def _make_named_partial(func, name, **kwargs):
    """Create a partial function with a proper __name__ attribute."""
    partial_func = partial(func, **kwargs)
    partial_func.__name__ = name
    return partial_func

bertscore_spanish = _make_named_partial(bertscore, "bertscore_spanish", language="es")
bertscore_english = _make_named_partial(bertscore, "bertscore_english", language="en")
code_bertscore_python = _make_named_partial(code_bertscore, "code_bertscore_python", code_lang="python")
code_bleu_python = _make_named_partial(code_bleu, "code_bleu_python", code_lang="python")

METRIC = {
    'anli': [accuracy],
    'common_gen': [rougeL, coverage],
    'squad': [squad],
    'cosmos_qa': [accuracy],
    'samsum': [rougeL, bertscore_english],
    'python_code': [code_bertscore_python, code_bleu_python],
    'xsum': [rougeL, bertscore_english],
    'bool_q': [accuracy],
    'eng_spa': [bertscore_spanish, sacrebleu],
    'paws': [accuracy],
    # 'quora': [bertscore_english],
    # 'alpaca': [bertscore_english]
}