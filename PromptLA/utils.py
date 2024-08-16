"""Some helper functions for PyTorch, including:
    - count_parameters: calculate parameters of network and display as a pretty table.
    - progress_bar: progress bar mimic xlua.progress.
"""
import csv
import os
import sys
import re
import time
import logging
import pickle
import shutil

import torch.nn as nn

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR
from torch.autograd import Variable
from torchvision.utils import save_image

from PIL import Image, ImageFont, ImageDraw

import matplotlib.pyplot as plt

term_width = 80
TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    """ creates progress bar for training"""
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    l = list()
    l.append('  Step: %s' % format_time(step_time))
    l.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        l.append(' | ' + msg)

    msg = ''.join(l)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def set_up_logger(file):
    # create custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # format for our loglines
    # formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # setup console logging
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # setup file logging as well
    fh = logging.FileHandler(file)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger