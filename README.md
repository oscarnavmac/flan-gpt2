# Distilling GPT-2 on Multi-Task Instructions

Distilling GPT-2 on a multi-task instruction dataset.

![knowledge-distillation](./images/tasks.png)
_Tasks collection used to finetune/distill GPT-2._

For now, we will only focus on fours tasks and later expand to the whole dataset.

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

For the T5-FLAN models, we should have expected to see a higher score for classification tasks. However we evaluated all models with the original dataset labels instead of the ones they were trained. For example, T5-FLAN trained on BoolQ using "yes" and "no" as targets instead of the original True and False. See the [evaluation notebook](https://colab.research.google.com/drive/1tfUkfX2p_CL7X7VqdHcrZxhlZErpMX3L) for more examples.

We cannot distill GPT-2 using FLAN-T5 as a teacher via KL-divergence (for example) because they have different tokenizer and vocabulary. Instead, we implemented [Universal Logit Distillation](https://arxiv.org/abs/2402.12030)