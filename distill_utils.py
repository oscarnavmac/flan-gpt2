from transformers import TrainingArguments, Trainer
import torch
import torch.nn as nn
import torch.nn.functional as F

class DistillationTrainingArguments(TrainingArguments):
    def __init__(self, *args, alpha=0.5, temperature=2.0, **kwargs):
        super().__init__(*args, **kwargs)

        self.alpha = alpha
        self.temperature = temperature

class DistillationTrainer(Trainer):
    def __init__(self, *args, teacher_model=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model
        self._move_model_to_device(self.teacher,self.model.device)
        self.teacher.eval()

    def compute_loss(self, model, inputs, return_outputs=False):

        # compute student output
        outputs_student = model(**inputs)
        student_loss=outputs_student.loss
        # compute teacher output
        with torch.no_grad():
          outputs_teacher = self.teacher(**inputs)

        assert outputs_student.logits.size() == outputs_teacher.logits.size()

        # Soften probabilities and compute distillation loss
        KL_Div = nn.KLDivLoss(reduction="batchmean")
        # Reversed Kullback-Leibler
        distill_loss = (KL_Div(
            F.log_softmax(outputs_teacher.logits / self.args.temperature, dim=-1),
            F.softmax(outputs_student.logits / self.args.temperature, dim=-1)) * (self.args.temperature ** 2))

        loss = self.args.alpha * student_loss + (1. - self.args.alpha) * distill_loss
        return (loss, outputs_student) if return_outputs else loss