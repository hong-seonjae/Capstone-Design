import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import SGRAF
from data import get_dataset, get_loader
from vocab import deserialize_vocab


def decode_tokens(token_ids, vocab):
    return " ".join([vocab.idx2word[str(idx)] for idx in token_ids])


def eval_loss(model, loader, data_size):
    model.val_start()
    losses = torch.zeros(data_size)

    with torch.no_grad():
        for images, captions, lengths, ids in tqdm(loader, desc="Evaluating losses", leave=False):
            loss = model.train(images, captions, lengths, mode="eval_loss")
            for j, idx in enumerate(ids):
                losses[idx] = loss[j]
    return losses


def plot_loss_distribution(output_dir="output", latest_model_dir="2025_04_07_15_16_52"):
    model_path = os.path.join(output_dir, latest_model_dir)
    print(f"Plotting based on: {model_path}")

    # Make directories for output
    loss_save_dir = os.path.join(model_path, "computed_loss")
    loss_plot_dir = os.path.join("plots", "loss_based")
    hard_text_dir = os.path.join("plots", "hard_negatives", "loss_based")

    os.makedirs(loss_save_dir, exist_ok=True)
    os.makedirs(loss_plot_dir, exist_ok=True)
    os.makedirs(hard_text_dir, exist_ok=True)

    # Search for model checkpoint files
    ckpt_files = sorted([
        f for f in os.listdir(model_path)
        if f.startswith("checkpoint_") and f.endswith(".pth.tar") and re.findall(r"\d+", f)
    ], key=lambda x: int(re.findall(r"\d+", x)[0]))
    ckpt_files = [f for f in ckpt_files if 1 <= int(re.findall(r"\d+", f)[0]) <= 30]

    # Load dataset, vocab, lookup dict
    checkpoint = torch.load(os.path.join(model_path, ckpt_files[0]), map_location="cpu")
    opt = checkpoint["opt"]
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, f"{opt.data_name}_vocab.json"))
    opt.vocab_size = len(vocab)

    captions_token, images = get_dataset(
        opt.data_path, opt.data_name, "train", vocab)
    loader, data_length, labels = get_loader(
        captions_token, images, "warmup", batch_size=128, workers=0,
        noise_ratio=opt.noise_ratio, noise_file=opt.noise_file
    )

    noisy_index = np.where(np.array(labels) == 0)[0]

    for ckpt_file in tqdm(ckpt_files, desc="Processing checkpoints"):
        epoch = int(re.findall(r"\d+", ckpt_file)[0])
        loss_path = os.path.join(loss_save_dir, f"loss_epoch_{epoch:02d}.npy")

        # Load precomputed loss if exists else compute loss
        if os.path.exists(loss_path):
            loss_A = np.load(loss_path)
        else:
            checkpoint = torch.load(os.path.join(model_path, ckpt_file))
  
            model = SGRAF(opt)
            model.load_state_dict(checkpoint["model_A"])
            model.val_start()

            print(f"🔁 [Epoch {epoch}] Computing loss...")
            loss_A = eval_loss(model, loader, data_length).numpy()
            np.save(loss_path, loss_A)
            print(f"✅ [Epoch {epoch}] Loss saved to {loss_path}")

        clean_loss = loss_A[np.array(labels) == 1]
        noisy_loss = loss_A[np.array(labels) == 0]

        # --- Loss based plot ---
        plt.figure(figsize=(8, 5))
        plt.hist(clean_loss, bins=200, alpha=0.6, label="Clean", color="blue", density=True)
        plt.hist(noisy_loss, bins=200, alpha=0.6, label="Noisy", color="red", density=True)
        plt.title(f"Raw Loss Distribution (Epoch {epoch})")
        plt.xlabel("Loss")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True)

        plt.savefig(os.path.join(loss_plot_dir, f"epoch_{epoch:02d}.png"))
        plt.close()

        # --- Hard Negative Extraction ---
        margin = 0.02
        median_clean = np.median(clean_loss)
        hard_negatives = [i for i in noisy_index if abs(loss_A[i] - median_clean) < margin]

        save_path = os.path.join(hard_text_dir, f"epoch_{epoch:02d}.txt")
        with open(save_path, "w") as f:
            f.write("📌 Hard negatives (noisy samples close to clean median):\n")
            for i in hard_negatives:
                noisy_img_id = loader.dataset._t2i_index[i]
                noisy_caption = decode_tokens(captions_token[i][1:-1], vocab)

                mismatched_img_id = loader.dataset.t2i_index[i]
                clean_caption_ids = [i for i in range(mismatched_img_id * 5, mismatched_img_id * 5 + 5)]

                f.write(f"[Caption #{i} for Image #{noisy_img_id} mismatched to Image #{mismatched_img_id}]: loss={loss_A[i]:.4f}\n")
                f.write(f"- Noisy Caption:\n\t({i}) {noisy_caption}\n")
                f.write(f"- Clean captions:\n")
                for j in clean_caption_ids:
                    clean_caption = decode_tokens(captions_token[j][1:-1], vocab)
                    f.write(f"\t({j}) {clean_caption}\n")
                f.write("\n")


if __name__ == "__main__":
    plot_loss_distribution()