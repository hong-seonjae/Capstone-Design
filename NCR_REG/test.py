import os
import sys
import time
import random
import logging
import argparse

import numpy as np
import torch

from utils import save_config, load_config
from evaluation import evalrank
from co_train_elr import main

project_dir = "/home/capstone_nc/NCR_REG"
data_dir = "/home/capstone_nc/NCR-data"
# model_ver = "ELR_2025_05_14_19_34_50"
models = ["ELR_2025_05_14_19_34_50", "ELR_2025_05_16_04_51_16"]
model_ver = models[1]

model_dir = os.path.join(project_dir, "output",  model_ver)
csv_dir = os.path.join(project_dir, "output", model_ver, "csv")
save_dir = os.path.join(project_dir, "plots", "elr_based", model_ver)
evalrank(os.path.join(model_dir, "checkpoints_train", "model_best.pth.tar"), split="test")
# evalrank(os.path.join(model_dir, "checkpoints_train", "checkpoint_14.pth.tar"), split="test")
