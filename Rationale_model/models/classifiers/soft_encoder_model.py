from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

import math
import numpy as np

from allennlp.data.dataset import Batch

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from Rationale_model.models.classifiers.base_model import RationaleBaseModel
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward
from allennlp.modules.attention import Attention
from Rationale_model.saliency_scorer.base_saliency_scorer import SaliencyScorer


@Model.register("soft_encoder_rationale_model")
class SoftEncoderRationaleModel(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        feedforward_encoder: FeedForward,
        attention: Attention,
        saliency_scorer: SaliencyScorer,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(SoftEncoderRationaleModel, self).__init__(vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._num_labels = self._vocabulary.get_vocab_size("labels")
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._dropout = torch.nn.Dropout(p=dropout)

        self._attention = attention

        self._feedforward_encoder = feedforward_encoder
        self._classifier_input_dim = self._feedforward_encoder.get_output_dim()

        self._num_labels = self._vocabulary.get_vocab_size("labels")
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self._num_labels)

        self._vector = torch.nn.Parameter(torch.randn((1, self._seq2seq_encoder.get_output_dim())))

        self.embedding_layers = [type(self._text_field_embedder)]

        self._saliency_scorer = saliency_scorer

        self._output_labels = self._vocabulary.get_index_to_token_vocabulary("labels")
        self._output_labels = [self._output_labels[i] for i in range(self._num_labels)]

        initializer(self)

    def forward(self, **kwargs):
        if not self.prediction_mode:
            return self._forward(**kwargs)
        else:
            scorer_dict = self._generate_attention(**kwargs)
            return scorer_dict

    def _forward(self, document, kept_tokens, rationale=None, label=None, metadata=None) -> Dict[str, Any]:
        embedded_text = self._text_field_embedder(document)
        mask = util.get_text_field_mask(document).float()

        embedded_text = self._dropout(self._seq2seq_encoder(embedded_text, mask=mask))
        attentions = self._attention(vector=self._vector, matrix=embedded_text, matrix_mask=mask)

        embedded_text = embedded_text * attentions.unsqueeze(-1) * mask.unsqueeze(-1)
        embedded_vec = self._feedforward_encoder(embedded_text.sum(1))

        logits = self._classification_layer(embedded_vec)
        probs = torch.nn.Softmax(dim=-1)(logits)

        output_dict = {}

        if label is not None:
            loss = F.cross_entropy(logits, label)
            output_dict["loss"] = loss

        output_dict["logits"] = logits
        output_dict["probs"] = probs
        output_dict["class_probs"] = probs.max(-1)[0]
        output_dict["predicted_labels"] = probs.argmax(-1)
        output_dict["gold_labels"] = label
        output_dict["metadata"] = metadata
        output_dict["attentions"] = attentions
        output_dict["mask"] = mask

        self._call_metrics(output_dict)

        return output_dict

    def _generate_attention(self, **kwargs) -> Dict[str, Any]:
        self._saliency_scorer.init_from_model(self)
        scorer_dict = self._saliency_scorer.score(**kwargs)
        return self._saliency_scorer.generate_comprehensiveness_metrics(scorer_dict, kwargs)

    def _decode(self, output_dict):
        new_output_dict = {}

        output_dict["predicted_labels"] = output_dict["predicted_labels"].cpu().data.numpy()

        masks = output_dict["mask"].cpu().data.numpy()
        metadata = output_dict["metadata"]
        soft_scores = output_dict["attentions"].cpu().data.numpy()

        new_output_dict["rationales"] = []

        for ss, mask, m in zip(soft_scores, masks, metadata):
            ss = ss[mask == 1]

            document_to_span_map = m["document_to_span_map"]
            document_rationale = []
            for docid, (s, e) in document_to_span_map.items():
                doc_ss = list(ss[s:e])
                doc_ss = [round(x, 8) if x == x else 0.0 for x in doc_ss]
                document_rationale.append({"docid": docid, "soft_rationale_predictions": doc_ss})

            new_output_dict["rationales"].append(document_rationale)

        output_labels = self._vocabulary.get_index_to_token_vocabulary("labels")

        new_output_dict["annotation_id"] = [m["annotation_id"] for m in metadata]
        new_output_dict["classification"] = [output_labels[int(p)] for p in output_dict["predicted_labels"]]
        new_output_dict["classification_scores"] = [
            dict(zip(self._output_labels, list(x))) for x in output_dict["probs"].cpu().data.numpy()
        ]

        new_output_dict["comprehensiveness_classification_scores"] = output_dict[
            "comprehensiveness_classification_scores"
        ]
        new_output_dict["sufficiency_classification_scores"] = output_dict["sufficiency_classification_scores"]
        new_output_dict["thresholded_scores"] = output_dict["thresholded_scores"]

        assert len(new_output_dict["classification"]) == len(new_output_dict["classification_scores"])

        return new_output_dict

    def label_array_to_dict(self, labels: np.ndarray):
        assert len(labels.shape) == 2
        return [dict(zip(self._output_labels, [float(y) for y in x])) for x in labels]

    def generate_tokens(self, new_tokens, metadata, labels):
        instances = []
        for tokens in new_tokens:
            instances += metadata[0]["convert_tokens_to_instance"](tokens, None)

        batch = Batch(instances)
        batch.index_instances(self._vocabulary)
        padding_lengths = batch.get_padding_lengths()

        batch = batch.as_tensor_dict(padding_lengths)
        return {k: v.to(self._vector.device) for k, v in batch["document"].items()}

    def regenerate_tokens(self, attentions, metadata, threshold, labels):
        attentions_cpu = attentions.cpu().data.numpy()
        sentences = [x["tokens"] for x in metadata]
        instances = []
        for b in range(attentions_cpu.shape[0]):
            sentence = [x for x in sentences[b]]
            always_keep_mask = metadata[b]['always_keep_mask']
            attn = attentions_cpu[b][: len(sentence)] + always_keep_mask * -10000
            max_length = math.ceil((1 - always_keep_mask).sum() * threshold)
            top_ind = np.argsort(attn)[-max_length:]
            new_tokens = [x for i, x in enumerate(sentence) if i in top_ind or always_keep_mask[i] == 1]
            instances += metadata[0]["convert_tokens_to_instance"](new_tokens, None)

        batch = Batch(instances)
        batch.index_instances(self._vocabulary)
        padding_lengths = batch.get_padding_lengths()

        batch = batch.as_tensor_dict(padding_lengths)
        return {k: v.to(attentions.device) for k, v in batch["document"].items()}

    def remove_tokens(self, attentions, metadata, threshold, labels):
        attentions_cpu = attentions.cpu().data.numpy()
        sentences = [x["tokens"] for x in metadata]
        instances = []
        for b in range(attentions_cpu.shape[0]):
            sentence = [x for x in sentences[b]]
            always_keep_mask = metadata[b]['always_keep_mask']
            attn = attentions_cpu[b][: len(sentence)] + always_keep_mask * -10000
            max_length = math.ceil((1 - always_keep_mask).sum() * threshold)

            top_ind = np.argsort(attn)[:-max_length]
            new_tokens = [x for i, x in enumerate(sentence) if i in top_ind or always_keep_mask[i] == 1]
            instances += metadata[0]["convert_tokens_to_instance"](new_tokens, None)

        batch = Batch(instances)
        batch.index_instances(self._vocabulary)
        padding_lengths = batch.get_padding_lengths()

        batch = batch.as_tensor_dict(padding_lengths)
        return {k: v.to(attentions.device) for k, v in batch["document"].items()}

