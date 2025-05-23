import os
import shutil
import json
import csv

import torch
import numpy as np

from scipy.stats import norm
from matplotlib import pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name="", fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def save_config(opt, file_path):
    with open(file_path, "w") as f:
        json.dump(opt.__dict__, f, indent=2)


def load_config(opt, file_path):
    with open(file_path, "r") as f:
        opt.__dict__ = json.load(f)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", prefix="", phase="train"):
    tries = 15
    error = None

    save_dir = os.path.join(prefix, f"checkpoints_{phase}")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, filename)
    best_path = os.path.join(save_dir, "model_best.pth.tar")

    while tries:
        try:
            torch.save(state, save_path)
            if is_best:
                shutil.copyfile(save_path, best_path)
        except IOError as e:
            error = e
            tries -= 1
            print(f"model save {filename} failed, remaining {tries} trials")
        else:
            break
    if not tries:
        raise error


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def write_file(opt, out="", option="validation", mode=None, model_name="A", epoch=None, stat_dict=None):
    if option == "validation":
        save_path = os.path.join(opt.output_dir, opt.save_validation_result)
    elif option == "loss":
        save_path = os.path.join(opt.output_dir, "loss", f"{mode}_loss_{model_name}_epoch{epoch+1}.txt")
        for img_id, cap_stats in stat_dict.items():
            out += f"Image {img_id}:\n"
            for s in cap_stats:
                out += (
                    f"\tCaption {s['caption_id']} | "
                    f"{'clean' if s['is_clean'] else 'noisy'} | "
                    f"raw similarity: {s['raw_similarity']:.3f}, "
                    f"similarity: {s['similarity']:.3f}, "
                    f"contrastive: {s['contrastive']:.3f}, "
                    f"elr: {s['elr']:.3f}\n"
                )
            out += "\n"

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(f"{save_path}", "a") as f:
        f.write(f"{out}\n")


def save_split_indices(opt, epoch, labeled_idx_A, unlabeled_idx_A, labeled_idx_B, unlabeled_idx_B):
    save_path = os.path.join(opt.output_dir, "split_indices", f"split_idx_epoch_{epoch}.npz")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    np.savez(
        save_path,
        labeled_A=labeled_idx_A,
        unlabeled_A=unlabeled_idx_A,
        labeled_B=labeled_idx_B,
        unlabeled_B=unlabeled_idx_B
    )

    print(
        f"A → labeled: {len(labeled_idx_A)}, unlabeled: {len(unlabeled_idx_A)}\n"
        f"B → labeled: {len(labeled_idx_B)}, unlabeled: {len(unlabeled_idx_B)}"
    )


def save_csv(opt, mode, model_name, epoch, stat_dict):
    csv_path = os.path.join(opt.output_dir, "csv", f"{mode}_{model_name}_epoch{epoch+1}.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["image_id", "caption_id", "is_clean", "similarity", "contrastive", "elr"])

        for img_id, cap_stats in stat_dict.items():
            for s in cap_stats:
                writer.writerow([
                    img_id,
                    s["caption_id"],
                    int(s["is_clean"]),
                    round(s["raw_similarity"], 3),
                    round(s["similarity"], 3),
                    round(s["contrastive"], 3),
                    round(s["elr"], 3)
                ])


# 단방향 하드 네거티브 기반 소프트 라벨 계산
def adaptive_prediction_hard_negative(S_it, S_i_all, temperature=0.07):
    """
    Args:
        S_it (Tensor): similarity of each positive pair, shape [B]
        S_i_all (Tensor): similarity matrix [B, B]
    Returns:
        Tensor: soft label in [0, 1], shape [B]
    """
    B = S_it.size(0)
    mask = ~torch.eye(B, dtype=torch.bool, device=S_it.device)
    S_i_neg_max = S_i_all.masked_fill(~mask, float('-inf')).max(dim=1).values
    margin = S_it - S_i_neg_max
    P = torch.sigmoid(margin / temperature)
    return P


# 양방향 하드 네거티브 기반 소프트 라벨 계산
def adaptive_prediction_hard_negative_bidirectional(S_it_matrix, temperature=0.07):
    """
    수식 3.1 기반의 양방향 하드 네거티브 기반 소프트 라벨 계산 함수

    Args:
        S_it_matrix (Tensor): 이미지-텍스트 간 유사도 행렬 [B, B], 각 (i, j)는 이미지 i와 텍스트 j의 유사도
        temperature (float): soft label을 스케일링하는 온도 파라미터 (τ)

    Returns:
        Tensor: 각 정답쌍 (i, i)에 대한 soft label score 벡터 [B]

    추가 : margin = S_diag - 0.5 * (S_i2t_neg + S_t2i_neg) (1/2하는 거 추가)

    """
    B = S_it_matrix.size(0)

    # 자기 자신 (정답쌍) 제외한 나머지를 위한 마스크 [B, B]
    mask = ~torch.eye(B, dtype=torch.bool, device=S_it_matrix.device)

    # I → T 방향에서 각 이미지별 가장 유사한 오답 텍스트 (hardest negative)
    S_i2t_neg = S_it_matrix.masked_fill(~mask, float('-inf')).max(dim=1).values  # [B]

    # T → I 방향에서 각 텍스트별 가장 유사한 오답 이미지 (hardest negative)
    S_t2i_neg = S_it_matrix.masked_fill(~mask, float('-inf')).max(dim=0).values  # [B]

    # 정답쌍 (i, i)의 유사도 (diagonal)
    S_diag = S_it_matrix.diag()  # [B]

    # 수식 3.1: margin = 정답쌍 유사도 - 양방향 하드네거티브 평균
    margin = S_diag - 0.5 * (S_i2t_neg + S_t2i_neg)

    # soft label = sigmoid( margin / temperature )
    return torch.sigmoid(margin / temperature)  # [B]

