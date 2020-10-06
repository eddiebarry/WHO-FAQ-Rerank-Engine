import abc
import torch

from typing import List, Mapping, Tuple, Union, Iterable, Optional, Any
from dataclasses import dataclass
from transformers import PreTrainedTokenizer


TokenizerReturnType = Mapping[str, Union[torch.Tensor, List[int],
                                         List[List[int]],
                                         List[List[str]]]]



class Query:
    """Class representing a query.
    A query contains the query text itself and potentially other metadata.
    Parameters
    ----------
    text : str
        The query text.
    id : Optional[str]
        The query id.
    """
    def __init__(self, text: str, id: Optional[str] = None):
        self.text = text
        self.id = id


class Text:
    """Class representing a text to be reranked.
    A text is unspecified with respect to it length; in principle, it
    could be a full-length document, a paragraph-sized passage, or
    even a short phrase.
    Parameters
    ----------
    text : str
        The text to be reranked.
    metadata : Mapping[str, Any]
        Additional metadata and other annotations.
    score : Optional[float]
        The score of the text. For example, the score might be the BM25 score
        from an initial retrieval stage.
    """

    def __init__(self,
                 text: str,
                 score: Optional[float] = 0):
        self.text = text
        self.score = score


class QueryDocumentBatch:
    def __init__(self, query: Query, documents: List[Text], \
                 output: Optional[TokenizerReturnType] = None):
        self.query = query
        self.documents = documents
        self.output = output

    def __len__(self):
        return len(self.documents)

class TokenizerEncodeMixin:
    def __init__(self, tokenizer: PreTrainedTokenizer = None, \
                 tokenizer_kwargs = None):
        self.tokenizer = tokenizer
        self.tokenizer_kwargs = tokenizer_kwargs

    def encode(self, strings: List[str]) -> TokenizerReturnType:
        assert self.tokenizer and self.tokenizer_kwargs is not None, \
                'mixin used improperly'
        ret = self.tokenizer.batch_encode_plus(strings,
                                               **self.tokenizer_kwargs)
        ret['tokens'] = list(map(self.tokenizer.tokenize, strings))
        return ret

class AppendEosTokenizerMixin:
    tokenizer: PreTrainedTokenizer = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, strings: List[str]) -> TokenizerReturnType:
        assert self.tokenizer, 'mixin used improperly'
        return super().encode(
            [f'{x} {self.tokenizer.eos_token}' for x in strings])

class QueryDocumentBatchTokenizer(TokenizerEncodeMixin):
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 batch_size: int,
                 pattern: str = '{query} {document}',
                 **tokenizer_kwargs):
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.tokenizer_kwargs = tokenizer_kwargs
        self.pattern = pattern

    def traverse_query_document(
            self,
            batch_input: QueryDocumentBatch) -> Iterable[QueryDocumentBatch]:
        query = batch_input.query
        for batch_idx in range(0, len(batch_input), self.batch_size):
            docs = batch_input.documents[batch_idx:batch_idx + self.batch_size]
            outputs = self.encode([self.pattern.format(
                                        query=query.text,
                                        document=doc.text) for doc in docs])
            yield QueryDocumentBatch(query, docs, outputs)

class Reranker:
    """Class representing a reranker.
    A reranker takes a list texts and returns a list of texts non-destructively
    (i.e., does not alter the original input list of texts).
    """
    @abc.abstractmethod
    def rerank(self, query: Query, texts: List[Text]) -> List[Text]:
        """Reranks a list of texts with respect to a query.
         Parameters
         ----------
         query : Query
             The query.
         texts : List[Text]
             The list of texts.
         Returns
         -------
         List[Text]
             Reranked list of texts.
         """
        pass