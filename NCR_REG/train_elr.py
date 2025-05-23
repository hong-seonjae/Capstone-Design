import os
import time
import random
import argparse
from collections import defaultdict
import csv

import torch
import numpy as np

from data import get_loader, get_dataset
from model_elr import SGRAF
from vocab import deserialize_vocab
from utils import AverageMeter, ProgressMeter, save_checkpoint, adjust_learning_rate, write_file

def warmup(opt, model, train_loader, labels, epoch):

    print(f"\nEpoch [{epoch+1}/{opt.warmup_epoch}]")
    adjust_learning_rate(opt, model.optimizer, epoch)
    loss_meter = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, loss_meter])

    model.train_start()
    end = time.time()

    stats = defaultdict(list)

    for i, (images, captions, lengths, ids) in enumerate(train_loader):
        loss, sample_stats = model.train(
            images,
            captions,
            lengths,
            ids=ids,
            labels=labels,
            mode="warmup"
        )

        loss_meter.update(loss, images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.log_step == 0:
            progress.display(i)

        for s in sample_stats:
            cap_id = s["caption_id"]
            img_id = train_loader.dataset.t2i_index[cap_id]
            stats[img_id].append(s)

    stats_path = os.path.join(opt.output_dir, f"loss_epoch{epoch+1}.txt")
    with open(stats_path, "w") as f:
        for img_id, cap_stats in stats.items():
            f.write(f"Image {img_id}:\n")
            for s in cap_stats:
                f.write(
                    f"\tCaption {s['caption_id']} | "
                    f"{'clean' if s['is_clean'] else 'noisy'} | "
                    f"similarity: {s['similarity']:.6f}, "
                    f"contrastive: {s['contrastive']:.6f}, "
                    f"elr: {s['elr']:.6f}\n"
                )
            f.write("\n")

def train(opt, model, train_loader, labels, epoch):

    print(f"\nEpoch [{epoch+1}/{opt.num_epochs}]")
    adjust_learning_rate(opt, model.optimizer, epoch)
    loss_meter = AverageMeter("loss", ":.4e")
    batch_time = AverageMeter("batch", ":6.3f")
    data_time = AverageMeter("data", ":6.3f")
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, loss_meter])

    model.train_start()
    end = time.time()

    # logs
    stats = defaultdict(list)

    for i, (images, captions, lengths, ids) in enumerate(train_loader):
        data_time.update(time.time() - end)

        loss, sample_stats = model.train(
            images,
            captions,
            lengths,
            ids=ids,
            labels=labels,
            mode="train"
        )

        loss_meter.update(loss, images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.log_step == 0:
            progress.display(i)

        for s in sample_stats:
            cap_id = s["caption_id"]
            img_id = train_loader.dataset.t2i_index[cap_id]
            stats[img_id].append(s)

    stats_path = os.path.join(opt.output_dir, f"loss_epoch{epoch+1}.txt")
    with open(stats_path, "w") as f:
        for img_id, cap_stats in stats.items():
            f.write(f"Image {img_id}:\n")
            for s in cap_stats:
                f.write(
                    f"\tCaption {s['caption_id']} | "
                    f"{'clean' if s['is_clean'] else 'noisy'} | "
                    f"similarity: {s['similarity']:.6f}, "
                    f"contrastive: {s['contrastive']:.6f}, "
                    f"elr: {s['elr']:.6f}\n"
                )
            f.write("\n")
            
    os.makedirs(os.path.join(opt.output_dir, "csv"), exist_ok=True)
    csv_path = os.path.join(opt.output_dir, "csv", f"csv_epoch{epoch+1}.csv")
    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "caption_id", "is_clean", "similarity", "contrastive", "elr"])
        
        for img_id, cap_stats in stats.items():
            for s in cap_stats:
                writer.writerow([
                    img_id,
                    s["caption_id"],
                    int(s["is_clean"]),  # 1 for clean, 0 for noisy
                    round(s["similarity"], 6),
                    round(s["contrastive"], 6),
                    round(s["elr"], 6)
                ])

def validate(opt, val_loader, models):
    from evaluation import i2t, t2i, encode_data, shard_attn_scores

    if opt.data_name == "cc152k_precomp":
        per_captions = 1
    else:
        per_captions = 5

    sims_mean = 0
    for model in models:
        img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt.log_step)
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), per_captions)])
        sims_mean += shard_attn_scores(model, img_embs, cap_embs, cap_lens, opt, shard_size=1000)

    sims_mean /= len(models)
    r1, r5, r10, medr, meanr = i2t(img_embs.shape[0], sims_mean, per_captions)
    r1i, r5i, r10i, medri, meanri = t2i(img_embs.shape[0], sims_mean, per_captions)

    print(f"Image to Text: {r1:.1f}, {r5:.1f}, {r10:.1f}")
    print(f"Text to Image: {r1i:.1f}, {r5i:.1f}, {r10i:.1f}")

    return r1 + r5 + r10 + r1i + r5i + r10i, r1, r5, r10, r1i, r5i, r10i

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="/home/capstone_nc/NCR-data/data")
    parser.add_argument("--data_name", default="f30k_precomp")
    parser.add_argument("--vocab_path", default="/home/capstone_nc/NCR-data/vocab")

    current_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    parser.add_argument("--output_dir", default=os.path.join("output", f"ELR_{current_time}"))

    # Data and training
    parser.add_argument("--noise_file", default="")
    parser.add_argument("--noise_ratio", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning_rate", type=float, default=0.0002)
    parser.add_argument("--log_step", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--lr_update", type=int, default=30)

    # Model configuration
    parser.add_argument("--img_dim", type=int, default=2048)
    parser.add_argument("--word_dim", type=int, default=300)
    parser.add_argument("--embed_size", type=int, default=1024)
    parser.add_argument("--sim_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--bi_gru", action="store_true")
    parser.add_argument("--no_imgnorm", action="store_true")
    parser.add_argument("--no_txtnorm", action="store_true")
    parser.add_argument("--module_name", default="SGR")
    parser.add_argument("--sgr_step", type=int, default=3)

    # Optimization options
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--soft_margin", default="exponential")
    parser.add_argument("--warmup_epoch", type=int, default=10)
    parser.add_argument("--warmup_model_path", type=str, default="warmup_model_10.pth.tar")
    parser.add_argument("--p_threshold", type=float, default=0.5)
    parser.add_argument("--no_co_training", action="store_true")

    # Logging and device
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_validation_result", type=str, default="validation.txt")
    parser.add_argument("--save_loss_result", type=str, default="loss.txt")

    # ELR
    parser.add_argument("--elr_lambda", type=float, default=0.5)
    parser.add_argument("--elr_beta", type=float, default=0.7)

    opt = parser.parse_args()

    # GPU setup
    torch.cuda.set_device(opt.gpu)

    # Seed
    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)

    os.makedirs(opt.output_dir, exist_ok=True)

    print("load and process dataset ...")
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, f"{opt.data_name}_vocab.json"))
    opt.vocab_size = len(vocab)

    # Load dataset
    captions_train, images_train = get_dataset(opt.data_path, opt.data_name, "train", vocab)
    captions_dev, images_dev = get_dataset(opt.data_path, opt.data_name, "dev", vocab)

    # subset
    # train_len = 40000
    # captions_train, images_train = captions_train[:train_len], images_train[:train_len // 5]
    # dev_len = 500
    # captions_dev, images_dev = captions_dev[:dev_len], images_dev[:dev_len]
    
    train_loader, data_size, labels = get_loader(
        captions_train, images_train, "warmup", opt.batch_size, opt.workers,
        opt.noise_ratio, opt.noise_file
    )
    val_loader = get_loader(captions_dev, images_dev, "dev", opt.batch_size, opt.workers)

    # Build model
    model = SGRAF(opt)
    warmup_best = 0
    best_rsum = 0

    # Warm-up
    if opt.warmup_model_path:
        if os.path.isfile(opt.warmup_model_path):
            checkpoint = torch.load(opt.warmup_model_path)
            model.load_state_dict(checkpoint["model_A"])
            print(
                f"=> loaded warmup checkpoint '{opt.warmup_model_path}' (epoch {checkpoint['epoch']})"
            )

            # Validation
            print("\nValidation ...")
            model.val_start()
            rsum, r1, r5, r10, r1i, r5i, r10i = validate(opt, val_loader, [model])
            write_file(opt, f"[WARMUP CHECKPOINT] r1: {r1} | r5: {r5} | r10: {r10} | r1i: {r1i} | r5i: {r5i} | r10i: {r10i}")
        else:
            raise Exception(
                "=> no checkpoint found at '{}'".format(opt.warmup_model_path)
            )
    else:
        for epoch in range(opt.warmup_epoch):
            warmup(opt, model, train_loader, labels, epoch)
        
            # Validation
            print("\nValidation ...")
            model.val_start()
            rsum, r1, r5, r10, r1i, r5i, r10i = validate(opt, val_loader, [model])
            write_file(opt, f"warmup epoch: {epoch+1} | r1: {r1} | r5: {r5} | r10: {r10} | r1i: {r1i} | r5i: {r5i} | r10i: {r10i}")

            # Save model
            is_best = rsum > warmup_best
            warmup_best = max(rsum, warmup_best)
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "model_A": model.state_dict(),
                    "best_rsum": best_rsum,
                    "opt": opt,
                }, 
                is_best,
                filename=f"warmup_model_{epoch+1}.pth.tar",
                prefix=opt.output_dir + "/"
            )

    # Train
    for epoch in range(opt.num_epochs):
        train(opt, model, train_loader, labels, epoch)
        
        # Validation
        print("\nValidation ...")
        model.val_start()
        rsum, r1, r5, r10, r1i, r5i, r10i = validate(opt, val_loader, [model])
        write_file(opt, f"epoch: {epoch+1} | r1: {r1} | r5: {r5} | r10: {r10} | r1i: {r1i} | r5i: {r5i} | r10i: {r10i}")

        # Save model
        is_best = rsum > best_rsum
        best_rsum = max(rsum, best_rsum)
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model_A": model.state_dict(),
                "best_rsum": best_rsum,
                "opt": opt,
            },
            is_best,
            filename=f"checkpoint_{epoch+1}.pth.tar",
            prefix=opt.output_dir + "/"
        )

if __name__ == "__main__":
    main()