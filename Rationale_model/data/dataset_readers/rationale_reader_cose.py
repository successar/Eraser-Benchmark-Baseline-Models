from typing import Dict, List, Tuple

from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import MetadataField, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.tokenizers import Token
import numpy as np
from Rationale_model.data.dataset_readers.rationale_reader import RationaleReader

@DatasetReader.register("rationale_reader_cose")
class RationaleReaderCoSE(RationaleReader):
    @overrides
    def text_to_instance(
        self,
        annotation_id: str,
        documents: Dict[str, List[str]],
        rationales: Dict[str, List[Tuple[int, int]]],
        query: str,
        label: str = None,
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}

        tokens = []
        is_evidence = []

        document_to_span_map = {}
        document_to_span_map_whole = {}

        docwords = documents[list(documents.keys())[0]]
        query = query.split("[sep]")
        query = [x.strip() for x in query]

        for docid, docwords in documents.items():
            document_to_span_map_whole[docid] = (len(tokens), len(tokens) + len(docwords))
            tokens += [Token(word) for word in docwords]
            document_to_span_map[docid] = (len(tokens) - len(docwords), len(tokens))

            tokens.append(Token("[SEP]"))

            rationale = [0] * len(docwords)
            if docid in rationales:
                for s, e in rationales[docid]:
                    for i in range(s, e):
                        rationale[i] = 1

            is_evidence += rationale + [1]

        always_keep_mask = [1 if t.text.upper() == "[SEP]" else 0 for t in tokens]

        fields["document"] = TextField(tokens, self._token_indexers)
        fields["rationale"] = SequenceLabelField(
            is_evidence, sequence_field=fields["document"], label_namespace="evidence_labels"
        )
        fields["kept_tokens"] = SequenceLabelField(
            always_keep_mask, sequence_field=fields["document"], label_namespace="kept_token_labels"
        )

        metadata = {
            "annotation_id": annotation_id,
            "tokens": tokens,
            "document_to_span_map": document_to_span_map,
            "convert_tokens_to_instance": self.convert_tokens_to_instance,
            "document_to_span_map_whole" : document_to_span_map_whole,
            "always_keep_mask" : np.array(always_keep_mask)
        }

        fields["metadata"] = MetadataField(metadata)
        fields["label"] = MetadataField({k: v for k, v in zip(["A", "B", "C", "D", "E", "Label"], query + [label])})

        return Instance(fields)

    def convert_tokens_to_instance(self, tokens, labels:List[str] = None):
        return [Instance({"document": TextField(tokens + [Token(l)], self._token_indexers)}) for l in labels]
