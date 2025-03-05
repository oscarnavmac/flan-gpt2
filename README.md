# Distilling GPT-2 on Multi-Task Instructions

In this project, we combine Kowledge Distillation (KD) alongside with instruction-tuning for training smaller language models (such as GPT-2) to follow natural multi-task prompts. We argue that models with less than a billion parameters are also capable to understand natural language prompts and generalize across 10 different tasks.

We cannot distill GPT-2 using FLAN-T5 as a teacher via KL-divergence (for example) because they have different tokenizer and vocabulary. Instead, we implemented [Universal Logit Distillation](https://arxiv.org/abs/2402.12030).

Our method demonstrated better performance compared with vanilla fine-tuning.

<figure>
<img src="./images/test_eval.png" width="400" />
<figcaption><i>Performance on our evaluation benchmark. The GPT-2 KD model shows better generalization than the FT version.</i></figcaption>
</figure>

## Tasks collection

For the training data, we used a custom subset of the Muffin collection (i.e. [FLAN_v1](https://arxiv.org/abs/2109.01652) dataset) in addition to more datasets. In this multi-task setting, we carefully selected a task collection composed of 10 Natural Language Understanding and Natural Language Generation datasets.

<figure>
<img src="./images/tasks.png" width="400" />
<figcaption><i>Tasks collection used to finetune/distill GPT-2.</i></figcaption>
</figure>

## Results

For now, we only focus on benchmarking with train dataset and comparing with teacher's family models.

| Model                             |  ANLI (acc) | BoolQ (acc) | CommonGen (rg-1) | XSum (rg-LSum) |
|-----------------------------------|-------------|-------------|------------------|----------------|
| google/flan-t5-small              | 0.008       | 0.2         | 0.275            | 0.175          |
| google/flan-t5-base               | 0.216       | 0.534       | 0.307            | 0.217          |
| google/flan-t5-large              | 0.298       | 0.634       | 0.333            | 0.254          |
| google/flan-t5-xl                 | 0.686       | 0.796       | 0.345            | 0.277          |
| google/flan-t5-xxl (quantized)    | 0.568       | 0.784       | 0.352            | 0.277          |
| google/flan-t5-small-distilled-xl | 0.32        | 0.564       | 0.120            | 0.150          |
| OscarNav/GPT-2-med-finetuned      | 0.308       | 0.37        | 0.247            | 0.131          |
| OscarNav/GPT-2-med-distilled      | 0.374       | 0.37        | 0.259            | 0.141          |

We compared results using our evaluation benchmarks, which consists of the dev (validation) and test split of the 8 white datasets. We used 200 examples for XSum and CoQA and 500 for the rest. We used rank classification to compute accuracy on ANLI, BoolQ, PAWS and CosmosQA. Only a single template was used for evaluation since we wanted to avoid instructions that could turn a task from NLU to NLG. The FLAN-T5-XXL was the larger model evaluated for his experiments but because it was validated using the same hardware that was used for training, we used 4-bit quantization to fit its weights into memory but loose some accuracy in the process. For the ANLI and PAWS datasets, our KD outperforms its teacher, presumably because the teacher was in disadvantage as it was trained with different ground-truth labels. This highlight the need to train with larger or finetune teachers to achieve better performance.

See the [evaluation notebook](https://colab.research.google.com/drive/1tfUkfX2p_CL7X7VqdHcrZxhlZErpMX3L) for more examples.

## Generalization on unseen tasks

For very few examples we actually saw some zero-shot behavior. 

![knowledge-distillation](./images/qualitative.png)

We argue that the model was able to generate a dialogue even though it was never explicitly trained with such task. Also in the QA, the model didn't return a translation or another question, but in return it responded properly.

