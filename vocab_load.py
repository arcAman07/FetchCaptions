from data.vocab import Vocab, build_vocab_from_file
from data.tokenizers import WordPunctTokenizer, CharTokenizer

LOAD_VOCABULARY = True #@param {type:"boolean"}
MIN_DF = 5 #@param {type:"integer"}

tokenizer_words = WordPunctTokenizer()
tokenizer_chars = CharTokenizer()

vocab_words = Vocab.load('vocab/vocab_words.txt')
vocab_chars = Vocab.load('vocab/vocab_chars.txt')

print('\nVocabulary sizes:')
print('WordVocab:', len(vocab_words))
print('CharVocab:', len(vocab_chars))