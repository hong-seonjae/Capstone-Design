import os
import re
import pickle

import torch
import numpy as np
from tqdm import tqdm

from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
from data import get_dataset, get_loader

def compute_loss(model_path, data_path=None, vocab_path=None, fold5=False):

    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint["opt"]

    opt.workers = 0

    if data_path is not None:
        opt.data_path = data_path
    if vocab_path is not None:
        opt.vocab_path = vocab_path

    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    # Load Vocabulary Wrapper
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
    )
    opt.vocab_size = len(vocab)

    if opt.data_name == "cc152k_precomp":
        captions, images, image_ids, raw_captions = get_dataset(
            opt.data_path, opt.data_name, "dev", vocab, return_id_caps=True
        )
    else:
        captions, images = get_dataset(opt.data_path, opt.data_name, "dev", vocab)
    data_loader = get_loader(captions=captions, images=images, data_split="dev", batch_size=opt.batch_size, workers=opt.workers)

    # construct model
    model_A = SGRAF(opt)
    model_B = SGRAF(opt)

    # load model state
    model_A.load_state_dict(checkpoint["model_A"])
    model_B.load_state_dict(checkpoint["model_B"])

    model_A.val_start()
    model_B.val_start()

    total_loss_A = 0
    total_loss_B = 0
    count = 0

    for images, captions, *_ in data_loader:
        images, captions = images.cuda(), captions.cuda()
        lengths = [cap.ne(0).sum().item() for cap in captions]

        loss_A = model_A.train(images, captions, lengths, mode="eval_loss")
        loss_B = model_B.train(images, captions, lengths, mode="eval_loss")

        total_loss_A += loss_A.sum().item()
        total_loss_B += loss_B.sum().item()
        count += images.size(0)
    
    total_loss = ((total_loss_A / count) + (total_loss_B / count)) / 2

    return total_loss

    


if __name__ == "__main__":

    model_path = os.path.expanduser("./output/2025_04_07_15_16_52/")
    data_path = os.path.expanduser("~/NCR-data/data/")
    vocab_path = os.path.expanduser("~/NCR-data/vocab/")

    model_name = [file for file in os.listdir(model_path) if file.startswith('checkpoint') and file.endswith('pth.tar')]
    model_name.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    result_list = []
    count = 0
    for model in tqdm(model_name):
        loss = compute_loss(
            os.path.join(model_path, model),
            data_path=data_path,
            vocab_path=vocab_path,
            fold5=False,
        )
        result_list.append({"epoch": count, "loss": loss})
        count += 1

    with open(os.path.join(model_path, 'train_loss_result.pkl'), 'wb') as f:
        pickle.dump(result_list, f)
