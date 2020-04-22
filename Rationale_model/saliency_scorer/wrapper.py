from Rationale_model.saliency_scorer.base_saliency_scorer import SaliencyScorer

@SaliencyScorer.register("wrapper")
class WrapperSaliency(SaliencyScorer) :    
    def score(self, **kwargs) :
        kept_tokens = kwargs['kept_tokens']
        output_dict = self._model['model']._forward(**kwargs)
        output_dict['attentions'] = output_dict['attentions'][:, :kept_tokens.shape[1]] * (1 - kept_tokens).float()
        output_dict['attentions'] = output_dict['attentions'] / output_dict['attentions'].sum(-1, keepdim=True)
        assert 'attentions' in output_dict, "No key 'attentions' in output_dict"
        return output_dict