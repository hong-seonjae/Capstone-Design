import os
import time
import copy
import shutil
import random
import json
import argparse

import numpy as np
import torch
from sklearn.mixture import GaussianMixture

from data import get_loader, get_dataset
from model import SGRAF
from vocab import Vocabulary, deserialize_vocab
from evaluation import i2t, t2i, encode_data, shard_attn_scores
from utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    adjust_learning_rate,
    write_file,
    save_split_indices
)


def main(opt):

    # load Vocabulary Wrapper
    print("load and process dataset ...")
    vocab = deserialize_vocab(
        os.path.join(opt.vocab_path, "%s_vocab.json" % opt.data_name)
    )
    opt.vocab_size = len(vocab)

    # load dataset
    captions_train, images_train = get_dataset(
        opt.data_path, opt.data_name, "train", vocab
    )
    captions_dev, images_dev = get_dataset(opt.data_path, opt.data_name, "dev", vocab)

    # data loader
    noisy_trainloader, data_size, clean_labels = get_loader(
        captions_train,
        images_train,
        "warmup",
        opt.batch_size,
        opt.workers,
        opt.noise_ratio,
        opt.noise_file,
    )
    val_loader = get_loader(
        captions_dev, images_dev, "dev", opt.batch_size, opt.workers
    )

    # create models
    model_A = SGRAF(opt)
    model_B = SGRAF(opt)

    checkpoint_path = os.path.join(opt.output_dir, 'model_best.pth.tar')
    checkpoint = torch.load(checkpoint_path)
    
    model_A.load_state_dict(checkpoint['model_A'])
    model_B.load_state_dict(checkpoint['model_B'])
    print("Checkpoint loaded.")

    best_rsum = 0
    start_epoch = 0

    # save the history of losses from two networks
    all_loss = [[], []]
    print("\n* Co-training with New Loss Function")

    # Train the Model
    for epoch in range(start_epoch, opt.num_epochs):
        print("\nEpoch [{}/{}]".format(epoch, opt.num_epochs))
        adjust_learning_rate(opt, model_A.optimizer, epoch)
        adjust_learning_rate(opt, model_B.optimizer, epoch)

        # # Dataset split (labeled, unlabeled)
        print("Split dataset ...")
        prob_A, prob_B, all_loss = eval_train(
            opt,
            model_A,
            model_B,
            noisy_trainloader,
            data_size,
            all_loss,
            clean_labels,
            epoch,
        )

        pred_A, _, _ = split_prob(prob_A, opt.p_threshold)
        pred_B, _, _ = split_prob(prob_B, opt.p_threshold)

        print("\nModel A training ...")
        # train model_A
        labeled_trainloader, _ = get_loader(
            captions_train,
            images_train,
            "train",
            opt.batch_size,
            opt.workers,
            opt.noise_ratio,
            opt.noise_file,
            pred=pred_B,
            prob=prob_B,
        )
        train(opt, model_A, model_B, labeled_trainloader, epoch)

        print("\nModel B training ...")
        # train model_B
        labeled_trainloader, _ = get_loader(
            captions_train,
            images_train,
            "train",
            opt.batch_size,
            opt.workers,
            opt.noise_ratio,
            opt.noise_file,
            pred=pred_A,
            prob=prob_A,
        )
        train(opt, model_B, model_A, labeled_trainloader, epoch)

        print("\nValidation ...")
        # evaluate on validation set
        rsum, r1, r5, r10, r1i, r5i, r10i = validate(opt, val_loader, [model_A, model_B])

        # save r1, r5, r10, r1i, r5i, r10i result
        write_file(opt, f"epoch: {epoch} | r1: {r1} | r5: {r5} | r10: {r10} | r1i: {r1i} | r5i: {r5i} | r10i: {r10i}")

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        if is_best:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_A": model_A.state_dict(),
                    "model_B": model_B.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best,
                filename="new_checkpoint_{}.pth.tar".format(epoch),
                prefix=opt.output_dir + "/",
            )
        else:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_A": model_A.state_dict(),
                    "model_B": model_B.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                },
                is_best=False,
                filename=f"new_checkpoint_{epoch}.pth.tar",
                prefix=opt.output_dir + "/"
            )


def train(opt, net, net2, labeled_trainloader, epoch=None):
    """
    One epoch training.
    """
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(labeled_trainloader),
        [batch_time, data_time, losses],
        prefix="Training Step",
    )

    # fix one network and train the other
    net.train_start()
    net2.val_start()

    end = time.time()
    for i, batch_train_data in enumerate(labeled_trainloader):
        pred_labels_l = []

        batch_images, batch_text, batch_lengths, _, batch_labels, batch_prob, batch_clean_labels = batch_train_data
        batch_size = batch_images.size(0)

        idx = torch.randperm(batch_size)
        random_images = batch_images[idx]
        random_text = batch_text[idx]
        random_lengths = [batch_lengths[i] for i in idx]

        # measure data loading time
        data_time.update(time.time() - end)

        # drop last batch if only one sample (batch normalization require)
        if batch_images.size(0) == 1:
            break

        net.train_start()
        # train with labeled + unlabeled data  exponential or linear
        loss_neg_images = net.train(
            batch_images,
            random_text,
            random_lengths,
            soft_margin=opt.soft_margin,
            mode="negative",
        )

        loss_neg_captions = net.train(
            random_images,
            batch_text,
            batch_lengths,
            soft_margin=opt.soft_margin,
            mode="negative"
        )

        loss = loss_neg_images + loss_neg_captions

        losses.update(loss, batch_images.size(0))
        write_file(opt, f"epoch: {epoch} | loss: {loss}", "loss")

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if i % opt.log_step == 0:
            progress.display(i)


def warmup(opt, train_loader, model, epoch):
    # average meters to record the training statistics
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses], prefix="Warmup Step"
    )

    end = time.time()
    for i, (images, captions, lengths, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # drop last batch if only one sample (batch normalization require)
        if images.size(0) == 1:
            break

        model.train_start()

        # Update the model
        loss = model.train(images, captions, lengths, mode="warmup")
        losses.update(loss, images.size(0))


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.log_step == 0:
            progress.display(i)


def validate(opt, val_loader, models=[]):
    # compute the encoding for all the validation images and captions
    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    elif opt.data_name in ["coco_precomp", "f30k_precomp"]:
        per_captions = 5

    Eiters = models[0].Eiters
    sims_mean = 0
    count = 0
    for ind in range(len(models)):
        count += 1
        print("Encoding with model {}".format(ind))
        img_embs, cap_embs, cap_lens = encode_data(
            models[ind], val_loader, opt.log_step
        )

        # clear duplicate 5*images and keep 1*images FIXME
        img_embs = np.array(
            [img_embs[i] for i in range(0, len(img_embs), per_captions)]
        )

        # record computation time of validation
        start = time.time()
        print("Computing similarity from model {}".format(ind))
        sims_mean += shard_attn_scores(
            models[ind], img_embs, cap_embs, cap_lens, opt, shard_size=1000
        )
        end = time.time()
        print(
            "Calculate similarity time with model {}: {:.2f} s".format(ind, end - start)
        )

    # average the sims
    sims_mean = sims_mean / count

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs.shape[0], sims_mean, per_captions)
    print(
        "Image to text: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1, r5, r10, medr, meanr
        )
    )

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(img_embs.shape[0], sims_mean, per_captions)
    print(
        "Text to image: {:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}".format(
            r1i, r5i, r10i, medri, meanr
        )
    )

    # sum of recalls to be used for early stopping
    r_sum = r1 + r5 + r10 + r1i + r5i + r10i

    return r_sum, r1, r5, r10, r1i, r5i, r10i


def eval_train(
    opt, model_A, model_B, data_loader, data_size, all_loss, clean_labels, epoch
):
    """
    Compute per-sample loss and prob
    """
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(data_loader), [batch_time, data_time], prefix="Computinng losses"
    )

    model_A.val_start()
    model_B.val_start()
    losses_A = torch.zeros(data_size)
    losses_B = torch.zeros(data_size)

    end = time.time()
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        with torch.no_grad():
            # compute the loss
            loss_A = model_A.train(images, captions, lengths, mode="eval_loss")
            loss_B = model_B.train(images, captions, lengths, mode="eval_loss")
            for b in range(images.size(0)):
                losses_A[ids[b]] = loss_A[b]
                losses_B[ids[b]] = loss_B[b]

            batch_time.update(time.time() - end)
            end = time.time()
            if i % opt.log_step == 0:
                progress.display(i)

    losses_A = (losses_A - losses_A.min()) / (losses_A.max() - losses_A.min())
    all_loss[0].append(losses_A)
    losses_B = (losses_B - losses_B.min()) / (losses_B.max() - losses_B.min())
    all_loss[1].append(losses_B)

    input_loss_A = losses_A.reshape(-1, 1)
    input_loss_B = losses_B.reshape(-1, 1)

    print("\nFitting GMM ...")
    # fit a two-component GMM to the loss
    gmm_A = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_A.fit(input_loss_A.cpu().numpy())
    prob_A = gmm_A.predict_proba(input_loss_A.cpu().numpy())
    prob_A = prob_A[:, gmm_A.means_.argmin()]

    gmm_B = GaussianMixture(n_components=2, max_iter=10, tol=1e-2, reg_covar=5e-4)
    gmm_B.fit(input_loss_B.cpu().numpy())
    prob_B = gmm_B.predict_proba(input_loss_B.cpu().numpy())
    prob_B = prob_B[:, gmm_B.means_.argmin()]

    return prob_A, prob_B, all_loss


def split_prob(prob, threshld):
    if prob.min() > threshld:
        # If prob are all larger than threshld, i.e. no noisy data, we enforce 1/100 unlabeled data
        print(
            "No estimated noisy data. Enforce the 1/100 data with small probability to be unlabeled."
        )
        threshld = np.sort(prob)[len(prob) // 100]
    pred = prob > threshld
    labeled_idx = np.where(pred)[0]
    unlabeled_idx = np.where(~pred)[0]
    return pred, labeled_idx, unlabeled_idx


if __name__ == '__main__':
    with open('./output/2025_04_12_12_08_21/config.json') as f:
        opt = argparse.Namespace(**json.load(f))

    main(opt)
