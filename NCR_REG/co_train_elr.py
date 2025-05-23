import os
import time
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

import torch
import numpy as np
from sklearn.mixture import GaussianMixture

from data import get_loader, get_dataset
from model_elr import SGRAF
from vocab import deserialize_vocab
from evaluation import i2t, t2i, encode_data, shard_attn_scores
from utils import (
    AverageMeter,
    ProgressMeter,
    save_config,
    save_checkpoint,
    adjust_learning_rate,
    write_file,
    save_split_indices,
    save_csv
)


def main(opt):

    print("load and process dataset ...")
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, f"{opt.data_name}_vocab.json"))
    opt.vocab_size = len(vocab)

    # Load dataset
    captions_train, images_train = get_dataset(opt.data_path, opt.data_name, "train", vocab)
    captions_dev, images_dev = get_dataset(opt.data_path, opt.data_name, "dev", vocab)

    # subset
    # train_len = 40000
    # captions_train, images_train = captions_train[:train_len], images_train[:train_len // 5]
    # dev_len = 1000
    # captions_dev, images_dev = captions_dev[:dev_len], images_dev[:dev_len]
    
    train_loader, data_size, gt_labels = get_loader(
        captions_train, images_train, "warmup", opt.batch_size, opt.workers, opt.noise_ratio, opt.noise_file
    )
    val_loader = get_loader(captions_dev, images_dev, "dev", opt.batch_size, opt.workers)

    # Build model
    model_A = SGRAF(opt)
    model_B = SGRAF(opt)

    best_rsum = 0

    # Warm-up
    print("\n* Warmup")
    if opt.warmup_model_path:
        if os.path.isfile(opt.warmup_model_path):
            checkpoint = torch.load(opt.warmup_model_path)
            model_A.load_state_dict(checkpoint["model_A"])
            model_B.load_state_dict(checkpoint["model_B"])
            print(
                f"=> loaded warmup checkpoint '{opt.warmup_model_path}' (epoch {checkpoint['epoch']})"
            )

            # Validation
            print("\nValidation ...")
            rsum, r1, r5, r10, r1i, r5i, r10i = validate(opt, val_loader, [model_A, model_B])
            write_file(opt, f"[WARMUP CHECKPOINT] r1: {r1} | r5: {r5} | r10: {r10} | r1i: {r1i} | r5i: {r5i} | r10i: {r10i}")
        else:
            raise Exception(
                "=> no checkpoint found at '{}'".format(opt.warmup_model_path)
            )
    else:
        for epoch in range(opt.warmup_epoch):
            print(f"\n[{epoch+1}/{opt.warmup_epoch}] Warmup model_A")
            warmup(opt, model_A, "A", train_loader, gt_labels, epoch)
            print(f"\n[{epoch+1}/{opt.warmup_epoch}] Warmup model_B")
            warmup(opt, model_B, "B", train_loader, gt_labels, epoch)

            model_A.val_start()
            model_B.val_start()

            stats_A = defaultdict(list)
            stats_B = defaultdict(list)
            for i, (images, captions, lengths, ids) in enumerate(train_loader):
                _, _, per_sample_stats_A = model_A.eval(
                    images, captions, lengths, gt_labels=gt_labels, mode="stat"
                )
                _, _, per_sample_stats_B = model_B.eval(
                    images, captions, lengths, gt_labels=gt_labels, mode="stat"
                )
                for s in per_sample_stats_A:
                    img_id = train_loader.dataset.t2i_index[s["caption_id"]]
                    stats_A[img_id].append(s)
                for s in per_sample_stats_B:
                    img_id = train_loader.dataset.t2i_index[s["caption_id"]]
                    stats_A[img_id].append(s)

            write_file(opt, option="loss", mode="warmup", model_name="A", epoch=epoch, stat_dict=stats_A)
            write_file(opt, option="loss", mode="warmup", model_name="B", epoch=epoch, stat_dict=stats_B)
            save_csv(opt, "warmup", "A", epoch, stats_A)
            save_csv(opt, "warmup", "A", epoch, stats_B)

            # Validation
            print("\nValidation ...")
            rsum, r1, r5, r10, r1i, r5i, r10i = validate(opt, val_loader, [model_A, model_B])
            write_file(opt, f"warmup epoch: {epoch+1} | r1: {r1} | r5: {r5} | r10: {r10} | r1i: {r1i} | r5i: {r5i} | r10i: {r10i}")

            # Save model
            warmup_best = max(rsum, warmup_best)
            save_checkpoint({
                "epoch": epoch+1,
                "model_A": model_A.state_dict(),
                "model_B": model_B.state_dict(),
                "opt": opt,
            }, is_best=False,filename=f"warmup_model_{epoch+1}.pth.tar", prefix=opt.output_dir + "/", phase="warmup")

    # Train
    print("\n* Co-training")

    for epoch in range(opt.num_epochs):
        print(f"\nEpoch [{epoch + 1}/{opt.num_epochs}]")
        adjust_learning_rate(opt, model_A.optimizer, epoch)
        adjust_learning_rate(opt, model_B.optimizer, epoch)

        # Sample selection
        prob_A, prob_B = eval_train(opt, model_A, model_B, train_loader, data_size, gt_labels)

        pred_A, labeled_idx_A, unlabeled_idx_A = split_prob(prob_A.numpy(), threshld=0.5)
        pred_B, labeled_idx_B, unlabeled_idx_B = split_prob(prob_B.numpy(), threshld=0.5)

        # Save split indices
        save_split_indices(opt, epoch, labeled_idx_A, unlabeled_idx_A, labeled_idx_B, unlabeled_idx_B)

        # train model A
        print("\nModel A training ...")
        labeled_loader_A, unlabeled_loader_A = get_loader(
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
        train(opt, model_A, model_B, labeled_loader_A, unlabeled_loader_A, epoch)

        # train model B
        print("\nModel B training ...")
        labeled_loader_B, unlabeled_loader_B = get_loader(
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
        train(opt, model_B, model_A, labeled_loader_B, unlabeled_loader_B, epoch)

        # Train statistics
        model_A.val_start()
        model_B.val_start()

        stats_A = defaultdict(list)
        stats_B = defaultdict(list)
        for i, (images, captions, lengths, ids) in enumerate(train_loader):
            _, _, per_sample_stats_A = model_A.eval(
                images, captions, lengths, ids=ids, gt_labels=gt_labels, mode="stat"
            )
            for s in per_sample_stats_A:
                img_id = train_loader.dataset.t2i_index[s["caption_id"]]
                stats_A[img_id].append(s)
    
            _, _, per_sample_stats_B = model_B.eval(
                images, captions, lengths, ids=ids, gt_labels=gt_labels, mode="stat"
            )
            for s in per_sample_stats_B:
                img_id = train_loader.dataset.t2i_index[s["caption_id"]]
                stats_B[img_id].append(s)

        write_file(opt, option="loss", mode="train", model_name="A", epoch=epoch, stat_dict=stats_A)
        save_csv(opt, "train", "A", epoch, stats_A)
        write_file(opt, option="loss", mode="train", model_name="B", epoch=epoch, stat_dict=stats_B)
        save_csv(opt, "train", "B", epoch, stats_B)

        # Validation
        print("\nValidation ...")
        rsum, r1, r5, r10, r1i, r5i, r10i = validate(opt, val_loader, [model_A, model_B])
        write_file(opt, f"epoch: {epoch+1} | r1: {r1} | r5: {r5} | r10: {r10} | r1i: {r1i} | r5i: {r5i} | r10i: {r10i}")

        # Save model
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            "epoch": epoch+1,
            "model_A": model_A.state_dict(),
            "model_B": model_B.state_dict(),
            "best_rsum": best_rsum,
            "opt": opt,
        }, is_best, filename=f"checkpoint_{epoch+1}.pth.tar", prefix=opt.output_dir + "/", phase="train")


def train(opt, net, net2, labeled_trainloader, unlabeled_trainloader, epoch):
    """
    One epoch training for one model (net), using predictions from the peer model (net2)
    """
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(labeled_trainloader),
        [batch_time, data_time, losses],
        prefix="Training Step",
    )

    net.train_start()
    net2.val_start()

    unlabeled_train_iter = iter(unlabeled_trainloader)
    labels_l = []
    pred_labels_l = []
    labels_u = []
    pred_labels_u = []
    end = time.time()
    for i, batch_train_data in enumerate(labeled_trainloader):
        (
            batch_images_l,
            batch_text_l,
            batch_lengths_l,
            batch_ids_l,
            batch_labels_l,
            batch_prob_l,
            batch_clean_labels_l,
        ) = batch_train_data
        batch_size = batch_images_l.size(0)
        labels_l.append(batch_clean_labels_l)

        # unlabeled data
        try:
            (
                batch_images_u,
                batch_text_u,
                batch_lengths_u,
                batch_ids_u,
                batch_clean_labels_u,
            ) = unlabeled_train_iter.next()
        except:
            unlabeled_train_iter = iter(unlabeled_trainloader)
            (
                batch_images_u,
                batch_text_u,
                batch_lengths_u,
                batch_ids_u,
                batch_clean_labels_u,
            ) = unlabeled_train_iter.next()
        labels_u.append(batch_clean_labels_u)

        # measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            batch_prob_l = batch_prob_l.cuda()
            batch_labels_l = batch_labels_l.cuda()

        # label refinement
        with torch.no_grad():
            net.val_start()
            # labeled data
            pl = net.predict(batch_images_l, batch_text_l, batch_lengths_l)
            ptl = batch_prob_l * batch_labels_l + (1 - batch_prob_l) * pl
            targets_l = ptl.detach()
            pred_labels_l.append(ptl.cpu().numpy())

            # unlabeled data
            pu1 = net.predict(batch_images_u, batch_text_u, batch_lengths_u)
            pu2 = net2.predict(batch_images_u, batch_text_u, batch_lengths_u)
            ptu = (pu1 + pu2) / 2
            targets_u = ptu.detach()
            targets_u = targets_u.view(-1, 1)
            pred_labels_u.append(ptu.cpu().numpy())

        # drop last batch if only one sample (batch normalization require)
        if batch_images_l.size(0) == 1 or batch_images_u.size(0) == 1:
            break

        net.train_start()
        # train with labeled + unlabeled data  exponential or linear
        loss_l = net.train(
            batch_images_l,
            batch_text_l,
            batch_lengths_l,
            ids=batch_ids_l,
            soft_labels=targets_l,
            soft_margin=opt.soft_margin,
            hard_negative=True,
            mode="train",
        )
        if epoch < (opt.num_epochs // 2):
            loss_u = 0
        else:
            loss_u = net.train(
                batch_images_u,
                batch_text_u,
                batch_lengths_u,
                ids=batch_ids_u,
                soft_labels=targets_u,
                hard_negative=True,
                soft_margin=opt.soft_margin,
                mode="train",
            )

        loss = loss_l + loss_u
        losses.update(loss, batch_images_l.size(0) + batch_images_u.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if i % opt.log_step == 0:
            progress.display(i)


def warmup(opt, model, model_name, train_loader, gt_labels, epoch):
    losses = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses], previx="Warmup Step"
    )

    end = time.time()
    for i, (images, captions, lengths, ids) in enumerate(train_loader):
        data_time.update(time.time() - end)

        # drop last batch if only one sample (batch normalization require)
        if images.size(0) == 1:
            break

        model.train_start()

        loss = model.train(
            images,
            captions,
            lengths,
            mode="warmup"
        )
        losses.update(loss, images.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.log_step == 0:
            progress.display(i)


def validate(opt, val_loader, models):
    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    else:
        per_captions = 5

    sims_mean = 0
    for model in models:
        img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step)
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])
        sims_mean += shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=100)

    sims_mean /= len(models)
    r1, r5, r10, medr, meanr = i2t(img_embs.shape[0], sims_mean, per_captions)
    r1i, r5i, r10i, medri, meanri = t2i(img_embs.shape[0], sims_mean, per_captions)

    print(f"Image to Text: {r1:.1f}, {r5:.1f}, {r10:.1f}")
    print(f"Text to Image: {r1i:.1f}, {r5i:.1f}, {r10i:.1f}")

    r_sum = r1 + r5 + r10 + r1i + r5i + r10i

    return r_sum, r1, r5, r10, r1i, r5i, r10i


def eval_train(opt, model_A, model_B, data_loader, data_size, gt_labels):
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
    
    contrasts_A = torch.zeros(data_size)
    contrasts_B = torch.zeros(data_size)
    elrs_A = torch.zeros(data_size)
    elrs_B = torch.zeros(data_size)

    gt_labels = torch.tensor(gt_labels).long()
    if torch.cuda.is_available():
        gt_labels = gt_labels.cuda()
        contrasts_A = contrasts_A.cuda()
        contrasts_B = contrasts_B.cuda()
        elrs_A = elrs_A.cuda()
        elrs_B = elrs_B.cuda()

    end = time.time()
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        contrast_A, elr_A, _ = model_A.eval(images, captions, lengths, ids=ids, gt_labels=gt_labels)
        contrast_B, elr_B, _ = model_B.eval(images, captions, lengths, ids=ids, gt_labels=gt_labels)

        contrasts_A[ids] = contrast_A
        contrasts_B[ids] = contrast_B
        elrs_A[ids] = elr_A
        elrs_B[ids] = elr_B

        batch_time.update(time.time() - end)
        end = time.time()
        if i % opt.log_step == 0:
            progress.display(i)

    # losses_A = contrasts_A + elrs_A
    # losses_B = contrasts_B + elrs_B
    losses_A = contrasts_A
    losses_B = contrasts_B
    losses_A = (losses_A - losses_A.min()) / (losses_A.max() - losses_A.min())
    losses_B = (losses_B - losses_B.min()) / (losses_B.max() - losses_B.min())

    print("\nFitting GMM ...")
    gmm_A = GaussianMixture(n_components=2, max_iter=100, tol=1e-3, reg_covar=1e-4)
    gmm_B = GaussianMixture(n_components=2, max_iter=100, tol=1e-3, reg_covar=1e-4)

    gmm_A.fit(losses_A.cpu().reshape(-1, 1).numpy())
    gmm_B.fit(losses_B.cpu().reshape(-1, 1).numpy())

    # Assign probabilities to the component with lower mean
    prob_A = gmm_A.predict_proba(losses_A.cpu().reshape(-1, 1).numpy())
    prob_B= gmm_B.predict_proba(losses_B.cpu().reshape(-1, 1).numpy())

    prob_A = prob_A[:, gmm_A.means_.argmin()]
    prob_B = prob_B[:, gmm_B.means_.argmin()]

    return torch.tensor(prob_A), torch.tensor(prob_B)



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