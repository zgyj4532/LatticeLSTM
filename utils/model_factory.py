# -*- coding: utf-8 -*-
from model.bilstmcrf import BiLSTM_CRF
from model.bilstm import BiLSTM


def build_seq_model(data, model_name):
    model_key = model_name.lower().replace('_', '').replace('+', '')
    if model_key == 'bilstmcrf':
        data.HP_use_char = False
        data.model_name = 'BiLSTM_CRF'
        data.char_model_name = 'None'
        return BiLSTM_CRF(data)
    if model_key == 'bilstm':
        data.model_name = 'BiLSTM'
        data.char_model_name = 'None'
        return BiLSTM(data)
    if model_key == 'charcnn':
        data.HP_use_char = True
        data.char_features = 'CNN'
        data.model_name = 'BiLSTM'
        data.char_model_name = 'CharCNN'
        return BiLSTM(data)
    if model_key == 'charbilstm':
        data.HP_use_char = True
        data.char_features = 'LSTM'
        data.model_name = 'BiLSTM'
        data.char_model_name = 'CharBiLSTM'
        return BiLSTM(data)
    raise ValueError(f'Unsupported model: {model_name}')