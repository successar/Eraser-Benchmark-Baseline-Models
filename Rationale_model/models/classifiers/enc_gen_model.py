from typing import Optional, Dict, Any

import torch
import torch.distributions as D

from allennlp.common.params import Params
from allennlp.data.vocabulary import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator, util
from Rationale_model.models.classifiers.base_model import RationaleBaseModel

from allennlp.training.metrics import Average


@Model.register("encoder_generator_rationale_model")
class EncoderGeneratorModel(RationaleBaseModel):
    def __init__(
        self,
        vocab: Vocabulary,
        generator: Params,
        encoder: Params,
        reg_loss_lambda: float,
        reg_loss_mu: float = 2,
        reinforce_loss_weight: float = 1.0,
        rationale_supervision_loss_weight: float = 1.0,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
    ):

        super(EncoderGeneratorModel, self).__init__(vocab, initializer, regularizer)
        self._vocabulary = vocab
        self._num_labels = self._vocabulary.get_vocab_size("labels")

        self._generator = Model.from_params(
            vocab=vocab, regularizer=regularizer, initializer=initializer, params=Params(generator)
        )
        self._encoder = Model.from_params(
            vocab=vocab, regularizer=regularizer, initializer=initializer, params=Params(encoder)
        )

        self._reg_loss_lambda = reg_loss_lambda
        self._reg_loss_mu = reg_loss_mu
        self._reinforce_loss_weight = reinforce_loss_weight
        self._rationale_supervision_loss_weight = rationale_supervision_loss_weight
        self._loss_tracks = {k: Average() for k in ["lasso_loss", "base_loss", "fused_lasso_loss"]}

        initializer(self)

    def forward(self, document, kept_tokens, rationale=None, label=None, metadata=None) -> Dict[str, Any]:
        generator_dict = self._generator(document, rationale)
        mask = util.get_text_field_mask(document)
        assert "prob_z" in generator_dict

        prob_z = generator_dict["prob_z"]
        assert len(prob_z.shape) == 2

        prob_z = kept_tokens.float() + prob_z * (1 - kept_tokens)
        sampler = D.bernoulli.Bernoulli(probs=prob_z)

        sample_z = sampler.sample() * mask.float()
        encoder_dict = self._encoder(sample_z=sample_z, label=label, metadata=metadata)

        loss = 0.0

        if label is not None:
            assert "loss" in encoder_dict

            loss_sample = encoder_dict["loss"]  # (B,)
            loss += loss_sample.mean()

            lasso_loss = util.masked_mean(sample_z, mask, dim=-1)  # (B,)

            masked_sum = mask[:, :-1].sum(-1).clamp(1e-5)
            diff = (sample_z[:, 1:] - sample_z[:, :-1]).abs()
            masked_diff = (diff * mask[:, :-1]).sum(-1)
            fused_lasso_loss = masked_diff / masked_sum

            self._loss_tracks["lasso_loss"](lasso_loss.mean().item())
            self._loss_tracks["fused_lasso_loss"](fused_lasso_loss.mean().item())
            self._loss_tracks["base_loss"](loss_sample.mean().item())

            log_prob_z = torch.log(1 + torch.exp(sampler.log_prob(sample_z)))  # (B, L)
            log_prob_z_sum = (mask * log_prob_z).mean(-1)  # (B,)

            generator_loss = (
                loss_sample.detach()
                + lasso_loss * self._reg_loss_lambda
                + fused_lasso_loss * (self._reg_loss_mu * self._reg_loss_lambda)
            ) * log_prob_z_sum

            loss += self._reinforce_loss_weight * generator_loss.mean()

        output_dict = generator_dict

        loss += self._rationale_supervision_loss_weight * generator_dict.get("rationale_supervision_loss", 0.0)

        output_dict["logits"] = encoder_dict["logits"]
        output_dict['probs'] = encoder_dict['probs']
        output_dict["class_probs"] = encoder_dict["class_probs"]
        output_dict["predicted_labels"] = encoder_dict["predicted_labels"]
        output_dict["gold_labels"] = encoder_dict["gold_labels"]

        output_dict["loss"] = loss
        output_dict["metadata"] = metadata
        output_dict["mask"] = mask

        self._call_metrics(output_dict)

        return output_dict

    def _decode(self, output_dict) -> Dict[str, Any]:
        new_output_dict = {}

        output_dict["predicted_labels"] = output_dict["predicted_labels"].cpu().data.numpy()

        masks = output_dict["mask"].cpu().data.numpy()
        predicted_rationales = output_dict["predicted_rationale"].cpu().data.numpy()
        metadata = output_dict["metadata"]
        soft_scores = output_dict["prob_z"].cpu().data.numpy()

        new_output_dict["rationales"] = []

        for rationale, ss, mask, m in zip(predicted_rationales, soft_scores, masks, metadata):
            rationale = rationale[mask == 1]
            ss = ss[mask == 1]

            document_to_span_map = m["document_to_span_map"]
            document_rationale = []
            for docid, (s, e) in document_to_span_map.items():
                doc_rationale = list(rationale[s:e]) + [0]
                starts = []
                ends = []
                for i in range(len(doc_rationale) - 1):
                    if (doc_rationale[i - 1], doc_rationale[i]) == (0, 1):
                        starts.append(i)
                    if (doc_rationale[i], doc_rationale[i + 1]) == (1, 0):
                        ends.append(i + 1)

                spans = zip(starts, ends)
                document_rationale.append(
                    {
                        "docid": docid,
                        "hard_rationale_predictions": [{"start_token": s, "end_token": e} for s, e in list(spans)],
                    }
                )

            new_output_dict["rationales"].append(document_rationale)

        output_labels = self._vocabulary.get_index_to_token_vocabulary("labels")

        new_output_dict["annotation_id"] = [m["annotation_id"] for m in metadata]
        new_output_dict["classification"] = [output_labels[int(p)] for p in output_dict["predicted_labels"]]

        _output_labels = [output_labels[i] for i in range(self._num_labels)]
        new_output_dict["classification_scores"] = [
            dict(zip(_output_labels, list(x))) for x in output_dict["probs"].cpu().data.numpy()
        ]

        return new_output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        base_metrics = super(EncoderGeneratorModel, self).get_metrics(reset)

        loss_metrics = {"_total" + k: v._total_value for k, v in self._loss_tracks.items()}
        loss_metrics.update({k: v.get_metric(reset) for k, v in self._loss_tracks.items()})
        loss_metrics.update(base_metrics)
        loss_metrics.update(self._generator.get_metrics(reset))

        return loss_metrics