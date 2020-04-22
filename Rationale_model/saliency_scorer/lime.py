from Rationale_model.saliency_scorer.base_saliency_scorer import SaliencyScorer
import torch
from math import ceil
from allennlp.data.tokenizers import Token
import numpy as np

from lime.lime_text import LimeTextExplainer


@SaliencyScorer.register("lime")
class LimeSaliency(SaliencyScorer):
    def init_from_model(self, model):
        self._model = {"model": model}

        output_labels = self._model["model"]._vocabulary.get_token_to_index_vocabulary("labels").keys()
        self._explainer = LimeTextExplainer(class_names=output_labels, split_expression=" ", bow=False)

        self._model["model"].eval()

    def score(self, **kwargs):
        kept_tokens = kwargs['kept_tokens']
        self._model["model"].eval()
        output_dict_orig = self._model["model"]._forward(**kwargs)

        metadata = kwargs["metadata"]

        assert "convert_tokens_to_instance" in metadata[0], breakpoint()
        selection_tokens = metadata[0]["tokens"]
        always_keep_mask = metadata[0]['always_keep_mask']

        selection_tokens = [i for i, x in enumerate(always_keep_mask) if x != 1]

        num_features = ceil(self._threshold * len(selection_tokens))

        predicted_label = output_dict_orig["predicted_labels"][0].item()

        def predict_proba(text_list):
            filtered_tokens = [
                [int(t) for t in selected_tokens.split(" ") if t != "UNKWORDZ"] for selected_tokens in text_list
            ]

            probs = []
            for i in range(0, len(filtered_tokens), self._batch_size):
                new_tokens = filtered_tokens[i : i + self._batch_size]
                new_tokens = [[t for i, t in enumerate(metadata[0]['tokens']) if i in tlist or always_keep_mask[i] == 1] for tlist in new_tokens]
                document = self._model["model"].generate_tokens(
                    new_tokens=new_tokens, metadata=metadata, labels=[kwargs["label"][0] for _ in range(len(new_tokens))]
                )

                l = [kwargs["label"][0] for _ in range(len(new_tokens))]
                try :
                    output = self._model["model"]._forward(
                        document=document,
                        kept_tokens=kwargs["kept_tokens"],
                        rationale=kwargs["rationale"],
                        label=torch.cat([x.unsqueeze(0) for x in l]) if type(l[0]) != dict else l,
                        metadata=kwargs["metadata"],
                    )
                except :
                    breakpoint()
                probs.append(output["probs"].cpu().data.numpy())
            return np.concatenate(probs, axis=0)

        explanation = self._explainer.explain_instance(
            " ".join([str(i) for i in selection_tokens]),
            predict_proba,
            num_features=min(len(selection_tokens), num_features*2),
            labels=(predicted_label,),
            num_samples=1000,
        )

        weights = explanation.as_list(predicted_label)
        saliency = [0.0 for _ in range(len(metadata[0]["tokens"]))]
        for f, w in weights:
            saliency[int(f)] = max(0.0, w)

        saliency = torch.Tensor([saliency]).to(output_dict_orig["probs"].device)

        output_dict_orig["attentions"] = saliency
        output_dict_orig['attentions'] = output_dict_orig['attentions'] * (1 - kept_tokens).float()
        output_dict_orig['attentions'] = output_dict_orig['attentions'] / output_dict_orig['attentions'].sum(-1, keepdim=True)

        return output_dict_orig
