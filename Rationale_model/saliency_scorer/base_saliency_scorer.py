from allennlp.common.registrable import Registrable
import torch


class SaliencyScorer(Registrable):
    def __init__(self, threshold: float, batch_size: int):
        self._threshold = threshold
        self._batch_size = batch_size
        self._aopc_thresholds = [0.003, 0.01, 0.05, 0.1, 0.2, 0.5]

    def init_from_model(self, model):
        self._model = {"model": model}

    def generate_comprehensiveness_metrics(self, scorer_dict, inputs):
        with torch.no_grad():
            torch.cuda.empty_cache()
            document = self._model["model"].regenerate_tokens(
                scorer_dict["attentions"], inputs["metadata"], self._threshold, inputs["label"]
            )

            output_dict_threshold = self._model["model"]._forward(
                document=document,
                kept_tokens=inputs["kept_tokens"],
                rationale=inputs["rationale"],
                label=inputs["label"],
                metadata=inputs["metadata"],
            )

            scorer_dict["sufficiency_classification_scores"] = self._model["model"].label_array_to_dict(
                output_dict_threshold["probs"].cpu().data.numpy()
            )
            del output_dict_threshold

            torch.cuda.empty_cache()

            scorer_dict["sufficiency_aopc_scores"] = {}
            scorer_dict["comprehensiveness_aopc_scores"] = {}

            for aopc_t in self._aopc_thresholds:
                torch.cuda.empty_cache()
                document = self._model["model"].regenerate_tokens(
                    scorer_dict["attentions"], inputs["metadata"], aopc_t, inputs["label"]
                )

                output_dict_threshold = self._model["model"]._forward(
                    document=document,
                    kept_tokens=inputs["kept_tokens"],
                    rationale=inputs["rationale"],
                    label=inputs["label"],
                    metadata=inputs["metadata"],
                )

                scorer_dict["sufficiency_aopc_scores"][aopc_t] = self._model["model"].label_array_to_dict(
                    output_dict_threshold["probs"].cpu().data.numpy()
                )
                del output_dict_threshold

            torch.cuda.empty_cache()
            document = self._model["model"].remove_tokens(
                scorer_dict["attentions"], inputs["metadata"], self._threshold, inputs["label"]
            )

            output_dict_threshold = self._model["model"]._forward(
                document=document,
                kept_tokens=inputs["kept_tokens"],
                rationale=inputs["rationale"],
                label=inputs["label"],
                metadata=inputs["metadata"],
            )

            scorer_dict["comprehensiveness_classification_scores"] = self._model["model"].label_array_to_dict(
                output_dict_threshold["probs"].cpu().data.numpy()
            )
            del output_dict_threshold

            for aopc_t in self._aopc_thresholds:
                torch.cuda.empty_cache()
                document = self._model["model"].remove_tokens(
                    scorer_dict["attentions"], inputs["metadata"], aopc_t, inputs["label"]
                )

                output_dict_threshold = self._model["model"]._forward(
                    document=document,
                    kept_tokens=inputs["kept_tokens"],
                    rationale=inputs["rationale"],
                    label=inputs["label"],
                    metadata=inputs["metadata"],
                )

                scorer_dict["comprehensiveness_aopc_scores"][aopc_t] = self._model["model"].label_array_to_dict(
                    output_dict_threshold["probs"].cpu().data.numpy()
                )
                del output_dict_threshold

        scorer_dict["thresholded_scores"] = []
        for i in range(len(scorer_dict["comprehensiveness_classification_scores"])):
            doc_scores = []
            for k in self._aopc_thresholds:
                doc_scores.append(
                    {
                        "threshold": k,
                        "comprehensiveness_classification_scores": scorer_dict["comprehensiveness_aopc_scores"][k][i],
                        "sufficiency_classification_scores": scorer_dict["sufficiency_aopc_scores"][k][i],
                    }
                )

            scorer_dict["thresholded_scores"].append(doc_scores)

        return scorer_dict

    def score(self, **inputs):
        raise NotImplementedError
