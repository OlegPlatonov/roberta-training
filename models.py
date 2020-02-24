import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_roberta import RobertaModel, ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
from transformers.configuration_roberta import RobertaConfig
from transformers.modeling_bert import BertPreTrainedModel


class GTHead(nn.Module):
    def __init__(self, hidden_size):
        super(GTHead, self).__init__()
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
        self.gt_head = GTHead(config.hidden_size)

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
