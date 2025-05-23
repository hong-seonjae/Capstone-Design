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
from collections import defaultdict

def decode_tokens(token_ids, vocab):
    return " ".join([vocab.idx2word[str(idx)] for idx in token_ids])


def compute_similarity(model, loader, data_length):
    model.val_start()
    sims = torch.zeros(data_length)

    with torch.no_grad():
        for images, captions, lengths, ids in tqdm(loader, desc="🔁 Computing similarities"):
            sim = model.forward_sim(*model.forward_emb(images, captions, lengths))
            sims[ids] = sim.diag().detach().cpu()
    return sims.numpy()


def compute_noisy_similarity(model, loader, data_length, labels):
    model.val_start()
    noisy_sims = np.full(data_length, np.nan)

    with torch.no_grad():
        for images, captions, lengths, ids in tqdm(loader, desc="🔁 Computing noisy similarities"):
            noisy_mask = torch.tensor([labels[i] == 0 for i in ids])
            if noisy_mask.sum() == 0:
                continue

            noisy_ids = [ids[i] for i, m in enumerate(noisy_mask) if m]
            sims = model.forward_sim(*model.forward_emb(images, captions, lengths))
            
            for i, idx in enumerate(noisy_ids):
                sim = sims[i, i].item()
                noisy_sims[idx] = sim

    return noisy_sims


def plot_similarity_distribution(output_dir="output", latest_model_dir="2025_04_07_15_16_52"):
    model_path = os.path.join(output_dir, latest_model_dir)
    print(f"Plotting based on: {model_path}")

    # Make directories
    sim_save_dir = os.path.join(model_path, "computed_sim2")
    sim_plot_dir = os.path.join("plots", "similarity_based2")
    hard_text_dir = os.path.join("plots", "hard_negatives", "similarity_based2")

    os.makedirs(sim_save_dir, exist_ok=True)
    os.makedirs(sim_plot_dir, exist_ok=True)
    os.makedirs(hard_text_dir, exist_ok=True)

    # Find model checkpoints
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
    clean_loader, _, _ = get_loader(
        captions_token, images, "warmup", batch_size=128, workers=0, noise_ratio=0
    )
    noisy_loader, data_length, labels = get_loader(
        captions_token, images, "warmup", batch_size=128, workers=0,
        noise_ratio=opt.noise_ratio, noise_file=opt.noise_file
    )

    _t2i_index = clean_loader.dataset._t2i_index
    t2i_index = noisy_loader.dataset.t2i_index

    # Image to [list of original captions]
    _i2t_index = defaultdict(list)
    for caption_idx, image_idx in enumerate(_t2i_index):
        _i2t_index[image_idx].append(caption_idx)

    for ckpt_file in tqdm(ckpt_files, desc="Processing checkpoints"):
        epoch = int(re.findall(r"\d+", ckpt_file)[0])
        clean_sim_path = os.path.join(sim_save_dir, f"epoch_{epoch:02d}_clean.npy")
        noisy_sim_path = os.path.join(sim_save_dir, f"epoch_{epoch:02d}_noisy.npy")

        checkpoint = torch.load(os.path.join(model_path, ckpt_file))
        model = SGRAF(opt)
        model.load_state_dict(checkpoint["model_A"])

        # Load or compute clean similarities
        if os.path.exists(clean_sim_path):
            clean_sims = np.load(clean_sim_path)
        else:
            clean_sims = compute_similarity(model, clean_loader, data_length)
            np.save(clean_sim_path, np.array(clean_sims))
            print(f"✅ [Epoch {epoch}] Clean similarities saved to {clean_sim_path}")

        if os.path.exists(noisy_sim_path):
            noisy_sims = np.load(noisy_sim_path)
        else:
            noisy_sims = compute_noisy_similarity(model, noisy_loader, data_length, labels)
            np.save(noisy_sim_path, np.array(noisy_sims))
            print(f"✅ [Epoch {epoch}] Noisy similarities saved to {noisy_sim_path}")

        plt.figure(figsize=(8, 5))
        plt.hist(clean_sims, bins=100, alpha=0.6, label="Clean", color="blue")
        plt.hist(noisy_sims[labels==0], bins=100, alpha=0.6, label="Noisy", color="red")
        plt.title(f"Similarity Distribution (Epoch {epoch}) [True Noise]")
        plt.xlabel("Similarity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(sim_plot_dir, f"epoch_{epoch:02d}.png"))
        plt.close()

        # --- Negatives samples and its original captions ---
        noisy_index = np.where(np.array(labels) == 0)[0]
        noisy_index = sorted(noisy_index, key=lambda i: noisy_sims[i])

        save_txt_path = os.path.join(hard_text_dir, f"epoch_{epoch:02d}.txt")
        with open(save_txt_path, "w") as f:
            f.write(f"For all {len(noisy_index)} noisy samples, we extracted the noisy captions along with their corresponding clean captions.\n\n")

            for i in noisy_index:
                noisy_img_id = _t2i_index[i]
                noisy_caption_ids = _i2t_index.get(noisy_img_id, [])

                mismatched_img_id = t2i_index[i]
                clean_caption_ids = _i2t_index.get(mismatched_img_id, [])

                f.write(f"[Caption #{i} for Image #{noisy_img_id} mistmatched to Image #{mismatched_img_id}]\n")
                f.write(f"- Noisy Captions:\n")
                for j in noisy_caption_ids:
                    noisy_caption = decode_tokens(captions_token[j][1:-1], vocab)
                    sim_noisy = noisy_sims[j]
                    f.write(f"\t({j}) Sim={sim_noisy:.4f} | {noisy_caption}\n")

                f.write(f"\n- Clean Captions:\n")
                for j in clean_caption_ids:
                    clean_caption = decode_tokens(captions_token[j][1:-1], vocab)
                    sim_clean = clean_sims[j]
                    f.write(f"\t({j}) Sim={sim_clean:.4f} | {clean_caption}\n")
                f.write("\n")


if __name__ == "__main__":
    plot_similarity_distribution()