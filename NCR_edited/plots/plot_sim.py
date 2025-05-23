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
    return sims


def plot_similarity_distribution(output_dir="output", latest_model_dir="2025_04_07_15_16_52"):
    model_path = os.path.join(output_dir, latest_model_dir)
    print(f"Plotting based on: {model_path}")

    # Make directories for output
    sim_save_dir = os.path.join(model_path, "computed_sim")
    sim_plot_dir = os.path.join("plots", "similarity_based")
    hard_text_dir = os.path.join("plots", "hard_negatives", "similarity_based")

    os.makedirs(sim_save_dir, exist_ok=True)
    os.makedirs(sim_plot_dir, exist_ok=True)
    os.makedirs(hard_text_dir, exist_ok=True)

    # Find model checkpoint files
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
    _t2i_index = loader.dataset._t2i_index # correct correspondence
    t2i_index = loader.dataset.t2i_index # noisy correspondence

    # Image to [list of original captions]
    _i2t_index = defaultdict(list)
    for caption_idx, image_idx in enumerate(_t2i_index):
        _i2t_index[image_idx].append(caption_idx)

    # Clean, Noisy sample indices
    clean_index = np.where(np.array(labels) == 1)[0]
    noisy_index = np.where(np.array(labels) == 0)[0]
    # print(_t2i_index[:10])
    # print(t2i_index[:10])
    # print(clean_index[:10])
    # print(noisy_index[:10])

    for ckpt_file in tqdm(ckpt_files, desc="Processing checkpoints"):
        epoch = int(re.findall(r"\d+", ckpt_file)[0])
        sim_path = os.path.join(sim_save_dir, f"epoch_{epoch:02d}.npy")

        # Load precomputed similarity if exists else compute similarity 
        if os.path.exists(sim_path):
            similarities = np.load(sim_path)
        else:
            checkpoint = torch.load(os.path.join(model_path, ckpt_file))
            model = SGRAF(opt)
            model.load_state_dict(checkpoint["model_A"])

            similarities = compute_similarity(model, loader, data_length).numpy()
            np.save(sim_path, similarities)
            print(f"✅ [Epoch {epoch}] Similarity saved to {sim_path}")

        # --- Similarity based plot ---
        clean_sims = similarities[clean_index]
        noisy_sims = similarities[noisy_index]

        plt.figure(figsize=(8, 5))
        plt.hist(clean_sims, bins=100, alpha=0.6, label="Clean", color="blue")
        plt.hist(noisy_sims, bins=100, alpha=0.6, label="Noisy", color="red")
        plt.title(f"Similarity Distribution (Epoch {epoch}) [True Noise]")
        plt.xlabel("Similarity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(sim_plot_dir, f"epoch_{epoch:02d}.png"))
        plt.close()

        # --- Negatives samples and its original captions ---
        sorted_noisy_index = sorted(noisy_index, key=lambda i: similarities[i])

        save_txt_path = os.path.join(hard_text_dir, f"epoch_{epoch:02d}.txt")
        with open(save_txt_path, "w") as f:
            f.write(f"For all {len(noisy_index)} noisy samples, we extracted the noisy captions along with their corresponding clean captions.\n\n")

            for i in sorted_noisy_index:
                noisy_img_id = _t2i_index[i]
                noisy_caption_ids = _i2t_index.get(noisy_img_id, [])

                mismatched_img_id = t2i_index[i]
                clean_caption_ids = _i2t_index.get(mismatched_img_id, [])

                f.write(f"[Caption #{i} for Image #{noisy_img_id} mistmatched to Image #{mismatched_img_id}]\n")
                f.write(f"- Noisy Captions:\n")
                for j in noisy_caption_ids:
                    noisy_caption = decode_tokens(captions_token[j][1:-1], vocab)
                    sim_noisy = similarities[j]
                    f.write(f"\t({j}) Sim={sim_noisy:.4f} | {noisy_caption}\n")

                f.write(f"\n- Clean Captions:\n")
                for j in clean_caption_ids:
                    clean_caption = decode_tokens(captions_token[j][1:-1], vocab)
                    sim_clean = similarities[j]
                    f.write(f"\t({j}) Sim={sim_clean:.4f} | {clean_caption}\n")
                f.write("\n")


if __name__ == "__main__":
    plot_similarity_distribution()