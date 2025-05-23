import os
import shutil
import json

import torch
import numpy as np

from scipy.stats import norm
from matplotlib import pyplot as plt

# 평균값 및 현재값을 저장하고 업데이트하는 유틸 클래스
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

# 학습 진행 상태를 출력하는 유틸 클래스
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


# 설정(config)을 JSON 파일로 저장
def save_config(opt, file_path):
    with open(file_path, "w") as f:
        json.dump(opt.__dict__, f, indent=2)

# JSON 파일로부터 설정 불러오기
def load_config(opt, file_path):
    with open(file_path, "r") as f:
        opt.__dict__ = json.load(f)

# 학습 체크포인트 저장 (최고 성능 모델 백업 포함)
def save_checkpoint(state, is_best, filename="checkpoint.pth.tar", prefix=""):
    tries = 15
    error = None

    # deal with unstable I/O. Usually not necessary.
    while tries:
        try:
            torch.save(state, prefix + filename)
            if is_best:
                shutil.copyfile(prefix + filename, prefix + "model_best.pth.tar")
        except IOError as e:
            error = e
            tries -= 1
        else:
            break
        print("model save {} failed, remaining {} trials".format(filename, tries))
        if not tries:
            raise error


# 학습률 조정 함수
def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

# 학습 중 로그 및 결과를 파일로 저장
def write_file(opt, out, option="validation"):
    if option == "validation":
        save_path = os.path.join(opt.output_dir, opt.save_validation_result)
    else:
        save_path = os.path.join(opt.output_dir, opt.save_loss_result)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(f"{save_path}", "a") as f:
        f.write(f"{out}\n")


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


