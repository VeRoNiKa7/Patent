from nltk import TweetTokenizer, sent_tokenize
from ufal.udpipe import Pipeline, Model, ProcessingError
from models import ConllTree

class UDPipeParser:
    def __init__(self, model_path, objectify=None):
        self._model = Model.load(model_path)
        self._tokenizer = TweetTokenizer()
        self._objectify = objectify or ConllTree.parse

    def _preprocess(self, text):
        tokenized_sentences = [self._tokenizer.tokenize(t) for t in sent_tokenize(text)]
        htext = '\n'.join(' '.join(sentence) for sentence in tokenized_sentences)
        return htext

    def parse(self, text):
        parser = Pipeline(self._model, 'horizontal', Pipeline.DEFAULT, Pipeline.DEFAULT, 'conllu')
        htext = self._preprocess(text)
        error = ProcessingError()
        result = parser.process(htext, error)
        return self._objectify(result)