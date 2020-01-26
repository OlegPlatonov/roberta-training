import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import RobertaModel
from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_bert import BertPreTrainedModel


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base":
        "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large":
        "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli":
        "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    "distilroberta-base":
        "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    "roberta-base-openai-detector":
        "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    "roberta-large-openai-detector":
        "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class GT_Head(nn.Module):
    def __init__(self, hidden_size):
        super(GT_Head, self).__init__()
        self.features_2_scores = nn.Linear(hidden_size, 1)

    def forward(self, sequence_output, gap_ids):
        batch_size, seq_len, _ = sequence_output.shape
        device = sequence_output.device

        index_batch = torch.arange(batch_size).unsqueeze(1).to(device)
        gap_ids = torch.cat([torch.zeros((batch_size, 1)).to(device).long(), gap_ids], dim=1)
        gaps = sequence_output[index_batch, gap_ids]

        gap_scores = self.features_2_scores(gaps).squeeze(-1)

        return gap_scores


class RobertaForGappedText(BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = 'roberta'

    def __init__(self, config):
        super(RobertaForGappedText, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.gt_head = GT_Head(config.hidden_size)

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

        gap_scores = self.gt_head(sequence_output=sequence_output, gap_ids=gap_ids)

        outputs = (gap_scores,) + outputs[2:]

        if target_gaps is not None:
            loss = F.cross_entropy(input=gap_scores, target=target_gaps)
            outputs = (loss,) + outputs

        return outputs
