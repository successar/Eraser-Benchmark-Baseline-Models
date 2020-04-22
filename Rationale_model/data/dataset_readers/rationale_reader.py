import os
from typing import Dict, List, Tuple

from overrides import overrides
import numpy as np
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import LabelField, MetadataField, TextField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Token

from rationale_benchmark.utils import annotations_from_jsonl, load_flattened_documents, Evidence


@DatasetReader.register("rationale_reader")
class RationaleReader(DatasetReader):
    def __init__(
        self,
        token_indexers: Dict[str, TokenIndexer],
        max_sequence_length: int = None,
        keep_prob: float = 1.0,
        lazy: bool = False,
    ) -> None:
        super().__init__(lazy=lazy)
        self._max_sequence_length = max_sequence_length
        self._token_indexers = token_indexers

        self._keep_prob = keep_prob
        self._bert = "bert" in token_indexers

    def generate_document_evidence_map(self, evidences: List[List[Evidence]]) -> Dict[str, List[Tuple[int, int]]]:
        document_evidence_map = {}
        for evgroup in evidences:
            for evclause in evgroup:
                if evclause.docid not in document_evidence_map:
                    document_evidence_map[evclause.docid] = []
                document_evidence_map[evclause.docid].append((evclause.start_token, evclause.end_token))

        return document_evidence_map

    @overrides
    def _read(self, file_path):
        data_dir = os.path.dirname(file_path)
        annotations = annotations_from_jsonl(file_path)
        documents: Dict[str, List[str]] = load_flattened_documents(data_dir, docids=None)

        for _, line in enumerate(annotations):
            annotation_id: str = line.annotation_id
            evidences: List[List[Evidence]] = line.evidences
            label: str = line.classification
            query: str = line.query
            docids: List[str] = sorted(list(set([evclause.docid for evgroup in evidences for evclause in evgroup])))

            filtered_documents: Dict[str, List[str]] = dict([(d, documents[d]) for d in docids])
            document_evidence_map = self.generate_document_evidence_map(evidences)

            if label is not None:
                label = str(label)

            instance = self.text_to_instance(
                annotation_id=annotation_id,
                documents=filtered_documents,
                rationales=document_evidence_map,
                query=query,
                label=label,
            )
            if instance is not None:
                yield instance

    @overrides
    def text_to_instance(
        self,
        annotation_id: str,
        documents: Dict[str, List[str]],
        rationales: Dict[str, List[Tuple[int, int]]],
        query: str = None,
        label: str = None,
    ) -> Instance:  # type: ignore
        # pylint: disable=arguments-differ
        fields = {}

        tokens = []
        is_evidence = []

        document_to_span_map = {}
        always_keep_mask = []

        for docid, docwords in documents.items():
            document_tokens = [Token(word) for word in docwords]
            tokens += document_tokens
            document_to_span_map[docid] = (len(tokens) - len(docwords), len(tokens))

            always_keep_mask += [0] * len(document_tokens)

            tokens.append(Token("[SEP]"))

            always_keep_mask += [1]

            rationale = [0] * len(docwords)
            if docid in rationales:
                for s, e in rationales[docid]:
                    for i in range(s, e):
                        rationale[i] = 1

            is_evidence += rationale + [1]

        if query is not None and type(query) != list:
            query_words = query.split()
            tokens += [Token(word) for word in query_words]
            tokens.append(Token("[SEP]"))
            is_evidence += [1] * (len(query_words) + 1)
            always_keep_mask += [1] * (len(query_words) + 1)

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
            "always_keep_mask" : np.array(always_keep_mask)
        }

        fields["metadata"] = MetadataField(metadata)

        if label is not None:
            fields["label"] = LabelField(label, label_namespace="labels")

        return Instance(fields)

    def convert_tokens_to_instance(self, tokens, labels=None):
        return [Instance({"document": TextField(tokens, self._token_indexers)})]
