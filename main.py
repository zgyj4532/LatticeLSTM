# -*- coding: utf-8 -*-
# @Author: Jie
# @Date:   2017-06-15 14:11:08
# @Last Modified by:   Jie Yang,     Contact: jieynlp@gmail.com
# @Last Modified time: 2018-07-06 11:08:27

import sys
import argparse
import random
import torch
import numpy as np
from utils.data import Data
from utils.data_utils import data_initialization, load_data_setting
from utils.training import train, load_model_decode

seed_num = 100
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tuning with bi-directional LSTM-CRF")
    parser.add_argument("--embedding", help="Embedding for words", default="None")
    parser.add_argument(
        "--status",
        choices=["train", "test", "decode"],
        help="update algorithm",
        default="train",
    )
    parser.add_argument(
        "--model",
        choices=["bilstmcrf", "bilstm", "charcnn", "charbilstm"],
        default="bilstmcrf",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--savemodel", default="data/saved_model.lstmcrf")
    parser.add_argument(
        "--savedset", help="Dir of saved data setting", default="data/save.dset"
    )
    parser.add_argument("--train", default="ResumeNER/train.char.bmes")
    parser.add_argument("--dev", default="ResumeNER/dev.char.bmes")
    parser.add_argument("--test", default="ResumeNER/test.char.bmes")
    parser.add_argument("--seg", default="True")
    parser.add_argument("--extendalphabet", default="True")
    parser.add_argument("--raw")
    parser.add_argument("--loadmodel")
    parser.add_argument("--output")
    args = parser.parse_args()

    train_file = args.train
    dev_file = args.dev
    test_file = args.test
    raw_file = args.raw
    model_dir = args.loadmodel
    dset_dir = args.savedset
    output_file = args.output
    if args.seg.lower() == "true":
        seg = True
    else:
        seg = False
    status = args.status.lower()

    save_model_dir = args.savemodel
    gpu = torch.cuda.is_available()

    char_emb = "data/gigaword_chn.all.a2b.uni.ite50.vec"
    bichar_emb = None
    gaz_file = "data/ctb.50d.vec"
    # gaz_file = None
    # char_emb = None
    # bichar_emb = None

    print("CuDNN:", torch.backends.cudnn.enabled)
    # gpu = False
    print("GPU available:", gpu)
    print("Status:", status)
    print("Model:", args.model)
    print("Epochs:", args.epochs)
    print("Seg: ", seg)
    print("Train file:", train_file)
    print("Dev file:", dev_file)
    print("Test file:", test_file)
    print("Raw file:", raw_file)
    print("Char emb:", char_emb)
    print("Bichar emb:", bichar_emb)
    print("Gaz file:", gaz_file)
    if status == "train":
        print("Model saved to:", save_model_dir)
    sys.stdout.flush()

    if status == "train":
        data = Data()
        data.HP_gpu = gpu
        data.HP_iteration = args.epochs
        data.HP_batch_size = 1
        data.use_bigram = False
        data.gaz_dropout = 0.5
        data.norm_gaz_emb = False
        data.HP_fix_gaz_emb = False
        data.model_name = args.model
        data.char_model_name = "None"
        if args.model == "charcnn":
            data.HP_use_char = True
            data.char_features = "CNN"
            data.char_model_name = "CharCNN"
        elif args.model == "charbilstm":
            data.HP_use_char = True
            data.char_features = "LSTM"
            data.char_model_name = "CharBiLSTM"
        elif args.model == "bilstm":
            data.HP_use_char = False
            data.char_features = "CNN"
        else:
            data.HP_use_char = False
            data.char_features = "CNN"
        data_initialization(data, gaz_file, train_file, dev_file, test_file)
        data.train_file = train_file
        data.dev_file = dev_file
        data.test_file = test_file
        data.generate_instance_with_gaz(train_file, "train")
        data.generate_instance_with_gaz(dev_file, "dev")
        data.generate_instance_with_gaz(test_file, "test")
        data.build_word_pretrain_emb(char_emb)
        data.build_biword_pretrain_emb(bichar_emb)
        data.build_gaz_pretrain_emb(gaz_file)
        train(data, save_model_dir, seg, args.model)
    elif status == "test":
        data = load_data_setting(dset_dir)
        data.generate_instance_with_gaz(dev_file, "dev")
        load_model_decode(
            model_dir, data, "dev", gpu, seg, getattr(data, "model_name", args.model)
        )
        data.generate_instance_with_gaz(test_file, "test")
        load_model_decode(
            model_dir, data, "test", gpu, seg, getattr(data, "model_name", args.model)
        )
    elif status == "decode":
        data = load_data_setting(dset_dir)
        data.generate_instance_with_gaz(raw_file, "raw")
        decode_results = load_model_decode(
            model_dir, data, "raw", gpu, seg, getattr(data, "model_name", args.model)
        )
        data.write_decoded_results(output_file, decode_results, "raw")
    else:
        print("Invalid argument! Please use valid arguments! (train/test/decode)")
