from __future__ import print_function
import os
import sys
import time
import json
from itertools import chain
import re
import pickle

import torch
import numpy as np
from tqdm import tqdm

from vocab import Vocabulary, deserialize_vocab
from model import SGRAF
from collections import OrderedDict
from utils import AverageMeter, ProgressMeter, write_file
from data import get_dataset, get_loader
from evaluation import encode_data, shard_attn_scores, i2t, t2i


def evalrank(model_path, data_path=None, vocab_path=None, split="dev", fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """

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
            opt.data_path, opt.data_name, split, vocab, return_id_caps=True
        )
    else:
        captions, images = get_dataset(opt.data_path, opt.data_name, split, vocab)
    data_loader = get_loader(captions, images, split, opt.batch_size, opt.workers)

    # construct model
    model_A = SGRAF(opt)
    model_B = SGRAF(opt)

    # load model state
    model_A.load_state_dict(checkpoint["model_A"])
    model_B.load_state_dict(checkpoint["model_B"])

    with torch.no_grad():
        img_embs_A, cap_embs_A, cap_lens_A = encode_data(model_A, data_loader)
        img_embs_B, cap_embs_B, cap_lens_B = encode_data(model_B, data_loader)

    if not fold5:
        # no cross-validation, full evaluation FIXME
        img_embs_A = np.array(
            [img_embs_A[i] for i in range(0, len(img_embs_A), per_captions)]
        )
        img_embs_B = np.array(
            [img_embs_B[i] for i in range(0, len(img_embs_B), per_captions)]
        )

        sims_A = shard_attn_scores(
            model_A, img_embs_A, cap_embs_A, cap_lens_A, opt, shard_size=1000
        )
        sims_B = shard_attn_scores(
            model_B, img_embs_B, cap_embs_B, cap_lens_B, opt, shard_size=1000
        )
        sims = (sims_A + sims_B) / 2

        # bi-directional retrieval
        r, rt = i2t(img_embs_A.shape[0], sims, per_captions, return_ranks=True)
        ri, rti = t2i(img_embs_A.shape[0], sims, per_captions, return_ranks=True)

        ar = (r[0] + r[1] + r[2]) / 3
        ari = (ri[0] + ri[1] + ri[2]) / 3
        rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]

        return (
            checkpoint["epoch"],
            {
                "rsum": rsum,
                "r1": r[0],
                "r5": r[1],
                "r10": r[2],
                "r1i": ri[0],
                "r5i": ri[1],
                "r10i": ri[2]
            }
        )

    else:
        # 5fold cross-validation, only for MSCOCO
        # need to fix
        results = []
        for i in range(5):
            # 5fold split
            img_embs_shard_A = img_embs_A[i * 5000 : (i + 1) * 5000 : 5]
            cap_embs_shard_A = cap_embs_A[i * 5000 : (i + 1) * 5000]
            cap_lens_shard_A = cap_lens_A[i * 5000 : (i + 1) * 5000]

            img_embs_shard_B = img_embs_B[i * 5000 : (i + 1) * 5000 : 5]
            cap_embs_shard_B = cap_embs_B[i * 5000 : (i + 1) * 5000]
            cap_lens_shard_B = cap_lens_B[i * 5000 : (i + 1) * 5000]

            start = time.time()
            sims_A = shard_attn_scores(
                model_A,
                img_embs_shard_A,
                cap_embs_shard_A,
                cap_lens_shard_A,
                opt,
                shard_size=1000,
            )
            sims_B = shard_attn_scores(
                model_B,
                img_embs_shard_B,
                cap_embs_shard_B,
                cap_lens_shard_B,
                opt,
                shard_size=1000,
            )
            sims = (sims_A + sims_B) / 2
            end = time.time()
            print("calculate similarity time:", end - start)

            r, rt0 = i2t(
                img_embs_shard_A.shape[0], sims, per_captions=5, return_ranks=True
            )
            print("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % r)
            ri, rti0 = t2i(
                img_embs_shard_A.shape[0], sims, per_captions=5, return_ranks=True
            )
            print("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" % ri)

            if i == 0:
                rt, rti = rt0, rti0
            ar = (r[0] + r[1] + r[2]) / 3
            ari = (ri[0] + ri[1] + ri[2]) / 3
            rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
            print("rsum: %.1f ar: %.1f ari: %.1f" % (rsum, ar, ari))
            results += [list(r) + list(ri) + [ar, ari, rsum]]

        print("-----------------------------------")
        print("Mean metrics: ")
        mean_metrics = tuple(np.array(results).mean(axis=0).flatten())
        print("rsum: %.1f" % (mean_metrics[10] * 6))
        mean_i2t = (mean_metrics[0] + mean_metrics[1] + mean_metrics[2]) / 3
        print("Average i2t Recall: %.1f" % mean_i2t)
        print("Image to text: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[:5])
        mean_t2i = (mean_metrics[5] + mean_metrics[6] + mean_metrics[7]) / 3
        print("Average t2i Recall: %.1f" % mean_t2i)
        print("Text to image: %.1f %.1f %.1f %.1f %.1f" % mean_metrics[5:10])


if __name__ == "__main__":

    model_path = os.path.expanduser("./output/2025_04_07_15_16_52/")
    data_path = os.path.expanduser("~/NCR-data/data/")
    vocab_path = os.path.expanduser("~/NCR-data/vocab/")

    model_name = [file for file in os.listdir(model_path) if file.startswith('checkpoint') and file.endswith('pth.tar')]
    model_name.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    result_list = []
    
    for model in tqdm(model_name):
        result = evalrank(
            os.path.join(model_path, model),
            data_path=data_path,
            vocab_path=vocab_path,
            split="test",
            fold5=False,
        )
        result_list.append(result)

    with open(os.path.join(model_path, 'recall_result.pkl'), 'wb') as f:
        pickle.dump(result_list, f)
