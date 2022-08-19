import torch
import os
import sys
from transformers import AutoModel
from models.caption_models import CaptioningLSTM

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_and_build_model(ckpt_path, model_class):

    print(f'Building {model_class.__name__} model')
    model = model_class.from_pretrained(ckpt_path)
    print(f'Built and loaded {model_class.__name__} model from {ckpt_path}')
    print('# parameters:', count_parameters(model))

    return model
  

FILE_TO_CLASS = {
    'LSTMDecoderWords.best.pth': CaptioningLSTM,
    'LSTMDecoderChars.best.pth': CaptioningLSTM,
    'LSTMDecoderWithLabelsWords.best.pth': CaptioningLSTMWithLabels,
    'LSTMDecoderWithLabelsChars.best.pth': CaptioningLSTMWithLabels,
    'TransformerDecoderBaseWords.best.pth': CaptioningTransformerBase,
    'TransformerDecoderBaseChars.best.pth': CaptioningTransformerBase,
    'TransformerDecoderWords.best.pth': CaptioningTransformer,
    'TransformerDecoderChars.best.pth': CaptioningTransformer
}


ckpt_path = 'LSTMDecoderWords.best.pth'
model_class = FILE_TO_CLASS[ckpt_path]

w_lstm_model = load_and_build_model(ckpt_path, model_class)