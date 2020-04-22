from Rationale_model.saliency_scorer.base_saliency_scorer import SaliencyScorer
import torch
import time

@SaliencyScorer.register("random")
class RandomSaliency(SaliencyScorer) :    
    def score(self, **kwargs) :
        kept_tokens = kwargs['kept_tokens']
        output_dict = self._model['model']._forward(**kwargs)
        output_dict['attentions'] = output_dict['attentions'][:, :kept_tokens.shape[1]] * (1 - kept_tokens).float()
        seed = time.time()
        torch.manual_seed(seed)
        output_dict['attentions'] = torch.rand_like(output_dict['attentions'])

        return output_dict