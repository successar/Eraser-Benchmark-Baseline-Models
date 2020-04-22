from Rationale_model.saliency_scorer.base_saliency_scorer import SaliencyScorer
import torch
import logging

@SaliencyScorer.register("simple_gradient")
class GradientSaliency(SaliencyScorer) :  
    def init_from_model(self, model) :
        self._embedding_layer = {}
        self._model = {'model' : model}
        logging.info("Initialising from Model .... ")
        model = self._model['model']

        _embedding_layer = [
            x for x in list(model.modules()) if any(str(y) in str(type(x)) for y in model.embedding_layers)
        ]
        assert len(_embedding_layer) == 1

        self._embedding_layer['embedding_layer'] = _embedding_layer[0]

    def score(self, **kwargs) :
        kept_tokens = kwargs['kept_tokens']
        self._model['model'].eval()
        output_dict_orig = self._model['model']._forward(**kwargs)

        keys = list(output_dict_orig.keys())
        for k in keys :
            if k not in ['predicted_labels', 'probs'] :
                del output_dict_orig[k]
                
        with torch.enable_grad() :
            self._model['model'].train()
            for param in self._embedding_layer['embedding_layer'].parameters():
                param.requires_grad = True

            embeddings_list = []
            def forward_hook(module, inputs, output):  # pylint: disable=unused-argument
                embeddings_list.append(output)
                output.retain_grad()

            hook = self._embedding_layer['embedding_layer'].register_forward_hook(forward_hook)
            output_dict = self._model['model']._forward(**kwargs)

            hook.remove()
            assert len(embeddings_list) == 1
            embeddings = embeddings_list[0]

            predicted_class_probs = output_dict["probs"][
                torch.arange(output_dict["probs"].shape[0]), output_dict["predicted_labels"]
            ]  # (B, C)

            predicted_class_probs.sum().backward(retain_graph=True)

            gradients = ((embeddings * embeddings.grad).sum(-1).detach()).abs()
            gradients = gradients / gradients.sum(-1, keepdim=True)

            output_dict['attentions'] = gradients
            output_dict['attentions'] = output_dict['attentions'][:, :kept_tokens.shape[1]] * (1 - kept_tokens).float()
            output_dict['attentions'] = output_dict['attentions'] / output_dict['attentions'].sum(-1, keepdim=True)

        output_dict['predicted_labels'] = output_dict_orig['predicted_labels']
        output_dict['probs'] = output_dict_orig['probs']
        return output_dict