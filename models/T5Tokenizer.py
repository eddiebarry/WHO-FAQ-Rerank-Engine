from base import AppendEosTokenizerMixin, QueryDocumentBatchTokenizer

class T5BatchTokenizer(AppendEosTokenizerMixin, QueryDocumentBatchTokenizer):
    def __init__(self, *args, **kwargs):
        kwargs['pattern'] = 'Query: {query} Document: {document} Relevant:'
        kwargs['return_attention_mask'] = True
        kwargs['pad_to_max_length'] = True
        kwargs['return_tensors'] = 'pt'
        kwargs['max_length'] = 256
        kwargs['truncation']=True
        super().__init__(*args, **kwargs)