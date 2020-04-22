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


@Model.register("soft_encoder_rationale_model_cose")
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
        self._vocabulary.add_tokens_to_namespace(["A", "B", "C", "D", "E"], namespace="labels")

        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._dropout = torch.nn.Dropout(p=dropout)

        self._attention = attention

        self._feedforward_encoder = feedforward_encoder
        self._classifier_input_dim = self._feedforward_encoder.get_output_dim()

        self._num_labels = self._vocabulary.get_vocab_size("labels")
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, 1)

        self._vector = torch.nn.Parameter(torch.randn((1, self._seq2seq_encoder.get_output_dim())))

        self.embedding_layers = [type(self._text_field_embedder)]

        self._saliency_scorer = saliency_scorer

        self._output_labels = self._vocabulary.get_index_to_token_vocabulary("labels")
        self._output_labels = [self._output_labels[i] for i in range(self._num_labels)]

        initializer(self)

    def forward(self, **kwargs):
        if not self.prediction_mode:
            document = self.generate_tokens(
                [m["tokens"] for m in kwargs["metadata"]], kwargs["metadata"], kwargs["label"]
            )
            return self._forward(
                document=document,
                kept_tokens=kwargs["kept_tokens"],
                rationale=kwargs["rationale"],
                label=kwargs["label"],
                metadata=kwargs["metadata"],
            )
        else:
            document = self.generate_tokens(
                [m["tokens"] for m in kwargs["metadata"]], kwargs["metadata"], kwargs["label"]
            )
            scorer_dict = self._generate_attention( document=document,
                kept_tokens=kwargs["kept_tokens"],
                rationale=kwargs["rationale"],
                label=kwargs["label"],
                metadata=kwargs["metadata"],)
            return scorer_dict

    def _forward(self, document, kept_tokens, rationale=None, label=None, metadata=None) -> Dict[str, Any]:
        embedded_text = self._text_field_embedder(document)
        mask = util.get_text_field_mask(document).float()

        embedded_text = self._dropout(self._seq2seq_encoder(embedded_text, mask=mask))
        attentions = self._attention(vector=self._vector, matrix=embedded_text, matrix_mask=mask)

        embedded_text = embedded_text * attentions.unsqueeze(-1) * mask.unsqueeze(-1)
        embedded_vec = self._feedforward_encoder(embedded_text.sum(1))

        logits = self._classification_layer(embedded_vec)
        b = logits.shape[0] // 5
        logits = logits.view(b, 5, 1)
        logits = logits.squeeze(-1)
        probs = torch.nn.Softmax(dim=-1)(logits)

        output_dict = {}

        output_labels = self._vocabulary.get_token_to_index_vocabulary("labels")
        int_label = torch.LongTensor([output_labels[l["Label"]] for l in label]).to(logits.device)
        loss = F.cross_entropy(logits, int_label)
        output_dict["loss"] = loss

        output_dict["logits"] = logits
        output_dict["probs"] = probs
        output_dict["class_probs"] = probs.max(-1)[0]
        output_dict["predicted_labels"] = probs.argmax(-1)
        output_dict["gold_labels"] = int_label
        output_dict["metadata"] = metadata
        output_dict['attentions'] = attentions

        output_dict['mask'] = mask

        self._call_metrics(output_dict)

        return output_dict

    def _generate_attention(self, document, kept_tokens, rationale, label, metadata) -> Dict[str, Any]:
        # Stupid Callback based Design
        self._saliency_scorer.init_from_model(self)
        kept_tokens_saliency = kept_tokens.unsqueeze(1).repeat(1, 5, 1).view(-1, kept_tokens.shape[1])
        scorer_dict = self._saliency_scorer.score(document=document, kept_tokens=kept_tokens_saliency, rationale=rationale, label=label, metadata=metadata)
        tokens = [m['tokens'] for m in metadata]
        attentions = scorer_dict['attentions']
        if attentions.shape[0] == 5*len(tokens) :
            attentions = attentions.view(len(tokens), 5, attentions.shape[-1])
            attentions = attentions.mean(1)
        elif attentions.shape[0] == len(tokens) :
            attentions = attentions
        else :
            breakpoint()
        scorer_dict['attentions'] = attentions
        return self._saliency_scorer.generate_comprehensiveness_metrics(scorer_dict, {
            'document' : document, 'kept_tokens' : kept_tokens, 'rationale' : rationale, 'label' : label, 'metadata' : metadata
        })

    def _decode(self, output_dict):
        new_output_dict = {}

        output_dict["predicted_labels"] = output_dict["predicted_labels"].cpu().data.numpy()

        metadata = output_dict["metadata"]
        soft_scores = output_dict["attentions"].cpu().data.numpy()

        new_output_dict["rationales"] = []

        for ss, m in zip(soft_scores, metadata):
            # ss = ss[mask == 1]

            document_to_span_map_whole = m["document_to_span_map_whole"]
            document_to_span_map = m['document_to_span_map']
            document_rationale = []
            for docid, (s, e) in document_to_span_map.items():
                whole_length = document_to_span_map_whole[docid][1] - document_to_span_map_whole[docid][0]
                doc_length = document_to_span_map[docid][1] - document_to_span_map[docid][0]
                doc_ss = list(ss[s:e]) + [0.0 for _ in range(whole_length - doc_length)]
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
        for tokens, instance_labels in zip(new_tokens, labels):
            instances += metadata[0]["convert_tokens_to_instance"](
                tokens, [instance_labels[k] for k in ["A", "B", "C", "D", "E"]]
            )

        batch = Batch(instances)
        batch.index_instances(self._vocabulary)
        padding_lengths = batch.get_padding_lengths()

        batch = batch.as_tensor_dict(padding_lengths)
        return {k: v.to(self._vector.device) for k, v in batch["document"].items()}

    def regenerate_tokens(self, attentions, metadata, threshold, labels):
        attentions_cpu = attentions.cpu().data.numpy()
        sentences = [x["tokens"] for x in metadata]
        new_tokens = []

        for b in range(attentions_cpu.shape[0]):
            sentence = [x for x in sentences[b]]
            always_keep_mask = metadata[b]['always_keep_mask']
            attn = attentions_cpu[b][: len(sentence)] + always_keep_mask * -10000
            max_length = math.ceil((1 - always_keep_mask).sum() * threshold)
            top_ind = np.argsort(attn)[-max_length:]
            new_tokens.append([x for i, x in enumerate(sentence) if i in top_ind or always_keep_mask[i] == 1])

        return self.generate_tokens(new_tokens, metadata, labels)

    def remove_tokens(self, attentions, metadata, threshold, labels):
        attentions_cpu = attentions.cpu().data.numpy()
        sentences = [x["tokens"] for x in metadata]
        new_tokens = []

        for b in range(attentions_cpu.shape[0]):
            sentence = [x for x in sentences[b]]
            always_keep_mask = metadata[b]['always_keep_mask']
            attn = attentions_cpu[b][: len(sentence)] + always_keep_mask * -10000
            max_length = math.ceil((1 - always_keep_mask).sum() * threshold)

            top_ind = np.argsort(attn)[:-max_length]
            new_tokens.append([x for i, x in enumerate(sentence) if i in top_ind or always_keep_mask[i] == 1])

        return self.generate_tokens(new_tokens, metadata, labels)

