from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder

from allennlp.training.metrics import F1Measure, Average

@Model.register("simple_generator_model")
class SimpleGeneratorModel(Model):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        feedforward_encoder: Seq2SeqEncoder,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(SimpleGeneratorModel, self).__init__(vocab, regularizer)
        self._vocabulary = vocab
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._dropout = torch.nn.Dropout(p=dropout)

        self._feedforward_encoder = feedforward_encoder
        self._classifier_input_dim = feedforward_encoder.get_output_dim()

        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, 1)

        self._rationale_f1_metric = F1Measure(positive_label=1)
        self._rationale_length = Average()
        self._rationale_supervision_loss = Average()

        initializer(self)

    def forward(self, document, rationale=None) -> Dict[str, Any]:
        embedded_text = self._text_field_embedder(document)
        mask = util.get_text_field_mask(document).float()

        embedded_text = self._dropout(self._seq2seq_encoder(embedded_text, mask=mask))
        embedded_text = self._feedforward_encoder(embedded_text)

        logits = self._classification_layer(embedded_text).squeeze(-1)
        probs = torch.sigmoid(logits)

        output_dict = {}

        predicted_rationale = (probs > 0.5).long()
        output_dict['predicted_rationale'] = predicted_rationale * mask
        output_dict["prob_z"] = probs * mask

        class_probs = torch.cat([1 - probs.unsqueeze(-1), probs.unsqueeze(-1)], dim=-1)

        average_rationale_length = util.masked_mean(output_dict['predicted_rationale'], mask, dim=-1).mean()
        self._rationale_length(average_rationale_length.item())

        if rationale is not None :
            rationale_loss = F.binary_cross_entropy_with_logits(logits, rationale.float(), weight=mask)
            output_dict['rationale_supervision_loss'] = rationale_loss
            output_dict['gold_rationale'] = rationale * mask
            self._rationale_f1_metric(predictions=class_probs, gold_labels=rationale, mask=mask)
            self._rationale_supervision_loss(rationale_loss.item())

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        p, r, f1 = self._rationale_f1_metric.get_metric(reset)
        metrics = {'_rationale_' + k:v for v, k in zip([p,r,f1], ['p', 'r', 'f1'])}
        metrics.update({'_rationale_length' : self._rationale_length.get_metric(reset)})
        metrics.update({'rationale_loss' : self._rationale_supervision_loss.get_metric(reset)})

        return metrics