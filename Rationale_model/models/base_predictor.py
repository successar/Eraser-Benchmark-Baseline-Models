from allennlp.predictors.predictor import Predictor
from typing import List
from allennlp.common.util import JsonDict, sanitize

from tqdm import tqdm

from allennlp.data import Instance
from allennlp.models import Model
from allennlp.data import DatasetReader

@Predictor.register("rationale_predictor")
class RationalePredictor(Predictor) :
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        self._model = model
        self._dataset_reader = dataset_reader
        self._tqdm = tqdm()

    def _json_to_instance(self, json_dict):
        raise NotImplementedError

    def predict_instance(self, instance: Instance) -> JsonDict:
        return self.predict_batch_instance([instance])[0]

    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        self._model.prediction_mode = True
        outputs = self._model.forward_on_instances(instances)
        self._tqdm.update(len(instances))
        return sanitize(outputs)