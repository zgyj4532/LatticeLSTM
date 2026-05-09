# -*- coding: utf-8 -*-
import gc
import json
import time

import torch

from utils.data_utils import batchify_with_label, predict_check, recover_label, save_data_setting, load_data_setting, lr_decay
from utils.metric import get_ner_fmeasure
from utils.model_factory import build_seq_model


def write_json_summary(summary_file, summary):
    with open(summary_file, 'w', encoding='utf-8') as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=2)
    print('Training summary saved to file: ', summary_file)


def evaluate(data, model, name):
    instances = []
    if name == "train":
        instances = data.train_Ids
    elif name == "dev":
        instances = data.dev_Ids
    elif name == 'test':
        instances = data.test_Ids
    elif name == 'raw':
        instances = data.raw_Ids
    else:
        print("Error: wrong evaluate name,", name)
    pred_results = []
    gold_results = []
    model.eval()
    batch_size = 1
    start_time = time.time()
    train_num = len(instances)
    total_batch = train_num // batch_size + 1
    for batch_id in range(total_batch):
        start = batch_id * batch_size
        end = (batch_id + 1) * batch_size
        if end > train_num:
            end = train_num
        instance = instances[start:end]
        if not instance:
            continue
        gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(instance, data.HP_gpu, True)
        tag_seq = model(gaz_list, batch_word, batch_biword, batch_wordlen, batch_char, batch_charlen, batch_charrecover, mask)
        pred_label, gold_label = recover_label(tag_seq, batch_label, mask, data.label_alphabet, batch_wordrecover)
        pred_results += pred_label
        gold_results += gold_label
    decode_time = time.time() - start_time
    speed = len(instances) / decode_time
    acc, p, r, f = get_ner_fmeasure(gold_results, pred_results, data.tagScheme)
    return speed, acc, p, r, f, pred_results


def train(data, save_model_dir, seg=True, model_name='bilstmcrf'):
    print("Training model...")
    data.show_data_summary()
    save_data_name = save_model_dir + ".dset"
    save_data_setting(data, save_data_name)
    model = build_seq_model(data, model_name)
    print("finished built model.")
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.SGD(parameters, lr=data.HP_lr, momentum=data.HP_momentum)
    best_dev = -1
    train_start_time = time.time()
    best_summary = None
    for idx in range(data.HP_iteration):
        epoch_start = time.time()
        temp_start = epoch_start
        print("Epoch: %s/%s" % (idx, data.HP_iteration))
        optimizer = lr_decay(optimizer, idx, data.HP_lr_decay, data.HP_lr)
        sample_loss = 0
        batch_loss = 0
        total_loss = 0
        right_token = 0
        whole_token = 0
        random = __import__('random')
        random.shuffle(data.train_Ids)
        model.train()
        model.zero_grad()
        batch_size = 1
        train_num = len(data.train_Ids)
        end = 0
        total_batch = train_num // batch_size + 1
        for batch_id in range(total_batch):
            start = batch_id * batch_size
            end = (batch_id + 1) * batch_size
            if end > train_num:
                end = train_num
            instance = data.train_Ids[start:end]
            if not instance:
                continue
            gaz_list, batch_word, batch_biword, batch_wordlen, batch_wordrecover, batch_char, batch_charlen, batch_charrecover, batch_label, mask = batchify_with_label(instance, data.HP_gpu)
            loss, tag_seq = model.neg_log_likelihood_loss(gaz_list, batch_word, batch_biword, batch_wordlen, batch_char, batch_charlen, batch_charrecover, batch_label, mask)
            right, whole = predict_check(tag_seq, batch_label, mask)
            right_token += right
            whole_token += whole
            sample_loss += loss.item()
            total_loss += loss.item()
            batch_loss += loss
            if end % 500 == 0:
                temp_time = time.time()
                temp_cost = temp_time - temp_start
                temp_start = temp_time
                print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
                sample_loss = 0
            if end % data.HP_batch_size == 0:
                batch_loss.backward()
                optimizer.step()
                model.zero_grad()
                batch_loss = 0
        temp_time = time.time()
        temp_cost = temp_time - temp_start
        print("     Instance: %s; Time: %.2fs; loss: %.4f; acc: %s/%s=%.4f" % (end, temp_cost, sample_loss, right_token, whole_token, (right_token + 0.) / whole_token))
        epoch_finish = time.time()
        epoch_cost = epoch_finish - epoch_start
        print("Epoch: %s training finished. Time: %.2fs, speed: %.2fst/s,  total loss: %s" % (idx, epoch_cost, train_num / epoch_cost, total_loss))
        speed, acc, p, r, f, _ = evaluate(data, model, "dev")
        dev_finish = time.time()
        dev_cost = dev_finish - epoch_finish
        if seg:
            current_score = f
            print("Dev: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (dev_cost, speed, acc, p, r, f))
        else:
            current_score = acc
            print("Dev: time: %.2fs speed: %.2fst/s; acc: %.4f" % (dev_cost, speed, acc))
        if current_score > best_dev:
            if seg:
                print("Exceed previous best f score:", best_dev)
            else:
                print("Exceed previous best acc score:", best_dev)
            model_path = save_model_dir + '.' + str(idx) + ".model"
            torch.save(model.state_dict(), model_path)
            best_dev = current_score
            best_summary = {
                "model": getattr(data, 'model_name', model_name),
                "char_model": getattr(data, 'char_model_name', 'None'),
                "epochs": int(data.HP_iteration),
                "best_epoch": int(idx),
                "best_score": float(best_dev),
                "time": float(dev_cost),
                "speed": float(speed),
                "acc": float(acc),
                "p": float(p),
                "r": float(r),
                "f": float(f),
                "loss": float(total_loss),
                "train_time": float(epoch_cost),
                "train_speed": float(train_num / epoch_cost),
                "model_path": model_path,
                "save_model_dir": save_model_dir,
                "train_file": getattr(data, 'train_file', None),
                "dev_file": getattr(data, 'dev_file', None),
                "test_file": getattr(data, 'test_file', None),
            }
        speed, acc, p, r, f, _ = evaluate(data, model, "test")
        test_finish = time.time()
        test_cost = test_finish - dev_finish
        if seg:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (test_cost, speed, acc, p, r, f))
        else:
            print("Test: time: %.2fs, speed: %.2fst/s; acc: %.4f" % (test_cost, speed, acc))
        if best_summary is not None and int(best_summary["best_epoch"]) == int(idx):
            best_summary["test_time"] = float(test_cost)
            best_summary["test_speed"] = float(speed)
            best_summary["test_acc"] = float(acc)
            best_summary["test_p"] = float(p)
            best_summary["test_r"] = float(r)
            best_summary["test_f"] = float(f)
        gc.collect()
    if best_summary is None:
        best_summary = {
            "model": getattr(data, 'model_name', model_name),
            "char_model": getattr(data, 'char_model_name', 'None'),
            "epochs": int(data.HP_iteration),
            "best_epoch": -1,
            "best_score": -1.0,
            "time": 0.0,
            "speed": 0.0,
            "acc": 0.0,
            "p": 0.0,
            "r": 0.0,
            "f": 0.0,
            "loss": 0.0,
            "train_time": 0.0,
            "train_speed": 0.0,
            "save_model_dir": save_model_dir,
            "model_path": None,
        }
    best_summary["total_time"] = float(time.time() - train_start_time)
    summary_file = save_model_dir + ".json"
    write_json_summary(summary_file, best_summary)
    return best_summary


def load_model_decode(model_dir, data, name, gpu, seg=True, model_name='bilstmcrf'):
    data.HP_gpu = gpu
    print("Load Model from file: ", model_dir)
    model = build_seq_model(data, model_name)
    model.load_state_dict(torch.load(model_dir))
    print("Decode %s data ..." % (name))
    start_time = time.time()
    speed, acc, p, r, f, pred_results = evaluate(data, model, name)
    end_time = time.time()
    time_cost = end_time - start_time
    if seg:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f, p: %.4f, r: %.4f, f: %.4f" % (name, time_cost, speed, acc, p, r, f))
    else:
        print("%s: time:%.2fs, speed:%.2fst/s; acc: %.4f" % (name, time_cost, speed, acc))
    return pred_results