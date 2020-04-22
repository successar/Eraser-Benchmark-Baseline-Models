from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from Rationale_model.models.classifiers.base_model import RationaleBaseModel
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder, FeedForward
from allennlp.modules.attention import Attention

from allennlp.data.dataset import Batch


@Model.register("encoder_rationale_model_cose")
class EncoderRationaleModel(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        text_field_embedder: TextFieldEmbedder,
        seq2seq_encoder: Seq2SeqEncoder,
        feedforward_encoder: FeedForward,
        attention: Attention,
        dropout: float = 0.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(EncoderRationaleModel, self).__init__(vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._vocabulary.add_tokens_to_namespace(["A", "B", "C", "D", "E"], namespace="labels")
        self._text_field_embedder = text_field_embedder
        self._seq2seq_encoder = seq2seq_encoder
        self._dropout = torch.nn.Dropout(p=dropout)

        self._attention = attention

        self._feedforward_encoder = feedforward_encoder
        self._classifier_input_dim = self._feedforward_encoder.get_output_dim()
        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, 1)

        self._vector = torch.nn.Parameter(torch.randn((1, self._seq2seq_encoder.get_output_dim())))

        self.embedding_layers = [type(self._text_field_embedder)]

        initializer(self)

    def forward(self, sample_z, label=None, metadata=None) -> Dict[str, Any]:
        document = self._regenerate_tokens_with_labels(metadata=metadata, sample_z=sample_z, labels=label)
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
        loss = F.cross_entropy(logits, int_label, reduction="none")
        output_dict["loss"] = loss

        output_dict["logits"] = logits
        output_dict["probs"] = probs
        output_dict["class_probs"] = probs.max(-1)[0]
        output_dict["predicted_labels"] = probs.argmax(-1)
        output_dict["gold_labels"] = int_label
        output_dict["metadata"] = metadata

        self._call_metrics(output_dict)

        return output_dict

    def _regenerate_tokens_with_labels(self, metadata, sample_z, labels):
        sample_z_cpu = sample_z.cpu().data.numpy()
        tokens = [m["tokens"] for m in metadata]

        assert len(tokens) == len(sample_z_cpu)
        assert max([len(x) for x in tokens]) == sample_z_cpu.shape[1]

        instances = []
        new_tokens = []
        for words, mask, meta, instance_labels in zip(tokens, sample_z_cpu, metadata, labels):
            mask = mask[: len(words)]
            new_words = [w for i, (w, m) in enumerate(zip(words, mask)) if i == 0 or m == 1]

            new_tokens.append(new_words)
            meta["new_tokens"] = new_tokens
            try :
                instances += metadata[0]["convert_tokens_to_instance"](
                    new_words, [instance_labels[k] for k in ["A", "B", "C", "D", "E"]]
                )
            except :
                breakpoint()

        batch = Batch(instances)
        batch.index_instances(self._vocabulary)
        padding_lengths = batch.get_padding_lengths()

        batch = batch.as_tensor_dict(padding_lengths)
        return {k: v.to(sample_z.device) for k, v in batch["document"].items()}
