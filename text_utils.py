import sys
from io import open
reload(sys)
sys.setdefaultencoding('utf-8')


class TextEncoder(object):
    """
    mostly a wrapper for a public python bpe tokenizer
    """

    def __init__(self, encoder_path):
        self.decoder = dict(enumerate([line.strip() for line in open(encoder_path,'r',encoding='utf-8')]))
        self.encoder = {v:k for k,v in self.decoder.items()}

    def encode(self, texts):
        texts_tokens = []
        for text in texts:
            text_tokens = [self.encoder.get(t, 0) for t in text.split(' ')]
            texts_tokens.append(text_tokens)
        return texts_tokens
