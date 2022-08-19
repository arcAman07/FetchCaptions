import torch
import os
import sys
from transformers import AutoModel
from models.caption_models import CaptioningLSTM
from PIL import Image
from experiments import text_to_seq, seq_to_text, split_caption
from imaging import memeify_image
import matplotlib.pyplot as plt
from data import MemeDataset
from PIL import Image
from experiments import text_to_seq, seq_to_text, split_caption
from imaging import memeify_image
from data.vocab import Vocab, build_vocab_from_file
from data.tokenizers import WordPunctTokenizer, CharTokenizer

DATA_DIR = 'memes900k'
CAPTIONS_FILE = os.path.join(DATA_DIR, 'captions_train.txt')

LOAD_VOCABULARY = True #@param {type:"boolean"}
MIN_DF = 5 #@param {type:"integer"}

tokenizer_words = WordPunctTokenizer()
tokenizer_chars = CharTokenizer()

if LOAD_VOCABULARY:
    vocab_words = Vocab.load('vocab/vocab_words.txt')
    vocab_chars = Vocab.load('vocab/vocab_chars.txt')
    print('Loaded vocabularies from Google Drive')
else:
    print(f'Building WordPunct Vocabulary from {CAPTIONS_FILE}, min_df={MIN_DF}')
    vocab_words = build_vocab_from_file(CAPTIONS_FILE, tokenizer_words, min_df=MIN_DF)

    print(f'Building Character Vocabulary from {CAPTIONS_FILE}, min_df={MIN_DF}')
    vocab_chars = build_vocab_from_file(CAPTIONS_FILE, tokenizer_chars, min_df=MIN_DF)


print('\nVocabulary sizes:')
print('WordVocab:', len(vocab_words))
print('CharVocab:', len(vocab_chars))

#@title Build `MemeDataset`


# use this to limit the dataset size (300 classes in total)
NUM_CLASSES = 200 #@param {type:"slider", min:1, max:300, step:1}  
PAD_IDX = vocab_words.stoi['<pad>']

from torchvision import transforms
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

splits = ['train', 'val', 'test']

datasets_words = {
    # WORD-LEVEL
    split: MemeDataset(DATA_DIR, vocab_words, tokenizer_words, image_transform=image_transform,
                       num_classes=NUM_CLASSES, split=split, preload_images=True)
    for split in splits
}

datasets_chars = {
    # CHAR-LEVEL
    split: MemeDataset(DATA_DIR, vocab_chars, tokenizer_chars, image_transform=image_transform,
                       num_classes=NUM_CLASSES, split=split, preload_images=True)
    for split in splits
}
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_and_build_model(ckpt_path, model_class):

    print(f'Building {model_class.__name__} model')
    model = model_class.from_pretrained(ckpt_path)
    print(f'Built and loaded {model_class.__name__} model from {ckpt_path}')
    print('# parameters:', count_parameters(model))

    return model
  

FILE_TO_CLASS = {
    'LSTMDecoderWords.best.pth': CaptioningLSTM
}


ckpt_path = 'LSTMDecoderWords.best.pth'
model_class = FILE_TO_CLASS[ckpt_path]

w_lstm_model = load_and_build_model(ckpt_path, model_class)

FONT_PATH = 'ImageCaptioning/fonts/impact.ttf'

def get_a_meme(model, img_torch, img_pil, caption, T=1., beam_size=7, top_k=50, 
               labels = None, mode = 'word'):
    if mode == 'word':
        vocabulary = vocab_words
        datasets = datasets_words
        delimiter=' '
        max_len = 32
    else:
        vocabulary = vocab_chars
        datasets = datasets_chars
        delimiter=''
        max_len = 128
    
    model.eval()
    if caption is not None:
        caption_tensor = torch.tensor(datasets['train']._preprocess_text(caption)[:-1]).unsqueeze(0)
    else:
        caption_tensor = None

    if labels is None:
        with torch.no_grad():
            output_seq = model.generate(
                image=img_torch, caption=caption_tensor,
                max_len=max_len, beam_size=beam_size, temperature=T, top_k=top_k
            )
    else:
        with torch.no_grad():
            output_seq = model.generate(
                image=img_torch, label=labels, caption=caption_tensor,
                max_len=max_len, beam_size=beam_size, temperature=T, top_k=top_k
            )
    
    pred_seq = output_seq
    text = seq_to_text(pred_seq, vocab=vocabulary, delimiter=delimiter)

    return text

# Image from dataset
label = 'PTSD Karate Kyle' 
labels = torch.tensor(datasets_words['train']._preprocess_text(label)).unsqueeze(0)
img_torch = datasets_words['train'].images[label]
img_pil = Image.open(datasets_words['train'].templates[label])
img_torch = img_torch.unsqueeze(0)
caption = None # "Your mom"

text = get_a_meme(
    model=w_lstm_model, T=1.3, 
    beam_size=10,
    top_k=100,
    img_torch=img_torch, 
    img_pil=img_pil, 
    caption=caption, 
    labels=None, 
    mode='word'
)

print(text)