from abc import ABC, ABCMeta

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_roberta import RobertaModel, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_bert import BertPreTrainedModel


class ModelRegistry(ABCMeta):
    registry = {}

    def __new__(mcs, name, bases, attrs):
        new_cls = ABCMeta.__new__(mcs, name, bases, attrs)
        mcs.registry[new_cls.task] = new_cls
        return new_cls

    @classmethod
    def get_model(mcs, task):
        return mcs.registry[task]


class BaseModel(ABC, BertPreTrainedModel, metaclass=ModelRegistry):
    task = None


class RobertaForGT(BaseModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'roberta'
    task = 'GT'

    def __init__(self, config):
        super(RobertaForGT, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.gap_outputs = nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        gap_ids,
        target_gaps=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]

        batch_size, seq_len, _ = sequence_output.shape
        device = sequence_output.device
        index_batch = torch.arange(batch_size).unsqueeze(1).to(device)
        gap_ids = torch.cat([torch.zeros((batch_size, 1)).to(device).long(), gap_ids], dim=1)
        gaps = sequence_output[index_batch, gap_ids]

        gap_logits = self.gap_outputs(gaps).squeeze(-1)

        outputs = (gap_logits,) + outputs[2:]

        if target_gaps is not None:
            loss = F.cross_entropy(input=gap_logits, target=target_gaps)

            all_losses = {'Loss': loss.item()}
            outputs = (loss, all_losses) + outputs

        return outputs


class RobertaForQA(BaseModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'roberta'
    task = 'QA'

    def __init__(self, config):
        super(RobertaForQA, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)

        self.init_weights()

    def forward(
        self,
        input_ids,
        attention_mask,
        answer_start=None,
        answer_end=None
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask
        )

        sequence_output = outputs[0]

        answer_logits = self.qa_outputs(sequence_output)

        start_logits, end_logits = answer_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits) + outputs[2:]

        if answer_start is not None and answer_end is not None:
            start_loss = F.cross_entropy(input=start_logits, target=answer_start)
            end_loss = F.cross_entropy(input=end_logits, target=answer_end)
            loss = start_loss + end_loss

            all_losses = {'Start_Loss': start_loss.item(), 'End_Loss': end_loss.item()}
            outputs = (loss, all_losses) + outputs

        return outputs
