import random
import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset
# from transformers import *
from transformers import BertTokenizer, XLNetTokenizer, get_linear_schedule_with_warmup, BertModel, BertConfig
from create_dataset import AVEC2014, calculate_label, DAIC_WOZ, SEARCH, calculate_labels_search
from numpy.random import randint
random.seed(336)


class MSADataset(Dataset):
    def __init__(self, config):
        self.mode = config.mode
        self.num_segments = 10
        self.duration = 2
        if "avec2014" in str(config.data_dir).lower():
            dataset = AVEC2014(config)
        if "daic-woz" in str(config.data_dir).lower():
            dataset = DAIC_WOZ(config)
        if "search" in str(config.data_dir).lower():
            dataset = SEARCH(config)

        self.data = dataset.get_data(config.mode)
        self.len = len(self.data)
        shample_number = dataset.get_shample_number(config.mode)
        # 计算权重（样本数的倒数）
        shample_number = torch.tensor(shample_number, dtype=torch.float32)
        weights = 1.0 / shample_number
        config.weights = weights / weights.sum()  # 正则化权重，使其和为 1
        config.visual_size = self.data[0][0].shape[1]
        config.acoustic_size = self.data[0][1].shape[1]
        # config.txt_size = self.data[0][0][2].shape[1]

    def _get_fragment(self, record):
        # # split all frames into seg parts, then select frame in each part
        # return record
        num_frames = record[0].shape[0]
        num_need_frames = self.num_segments * self.duration
        if num_need_frames > num_frames:
            offsets = list(range(num_frames))  # 返回0到X-1的所有数字
        # 从0到X-1中随机选择K个不同的数字
        else:
            # offsets = random.sample(range(num_frames), self.num_segments * self.duration)
            # offsets = sorted(offsets)

            step = num_frames / num_need_frames  # 均匀间隔
            # 确保均匀分布的同时添加随机偏移
            offsets = [int(i * step + random.uniform(0, step)) for i in range(num_need_frames)]
            offsets = [min(offset, num_frames - 1) for offset in offsets]  # 防止超出范围
            offsets = sorted(offsets)

        # 挑选帧
        visual_fea, audio_fea, number, score, label = record
        visual = visual_fea[offsets, :]
        audio = audio_fea[offsets, :]
        paragraph_inforamtion = (visual, audio, number, score, label)
        return paragraph_inforamtion

    def _get_fragment_test(self, record):
        # # split all frames into seg parts, then select frame in each part
        # return record
        num_frames = record[0].shape[0]
        num_need_frames = self.num_segments * self.duration
        if num_need_frames > num_frames:
            offsets = list(range(num_frames))  # 返回0到X-1的所有数字
        # 从0到X-1中随机选择K个不同的数字
        else:
            # offsets = random.sample(range(num_frames), self.num_segments * self.duration)
            # offsets = sorted(offsets)

            step = num_frames / num_need_frames  # 均匀间隔

            # 确保均匀分布的同时添加随机偏移
            offsets = [int(i * step) for i in range(num_need_frames)]
            offsets = [min(offset, num_frames - 1) for offset in offsets]  # 防止超出范围

            offsets = sorted(offsets)


        # 挑选帧
        visual_fea, audio_fea, number, score, label = record
        visual = visual_fea[offsets, :]
        audio = audio_fea[offsets, :]
        paragraph_inforamtion = (visual, audio, number, score, label)
        return paragraph_inforamtion


    def __getitem__(self, index):
        record = self.data[index]

        if self.mode == 'train':
            segment = self._get_fragment(record)
            return segment
        elif self.mode == 'test':
            segment = self._get_fragment(record)
            return segment
            # return record

        # return record


    def __len__(self):
        return self.len


def get_loader(config, shuffle=True):
    """Load DataLoader of given DialogDataset"""

    dataset = MSADataset(config)
    config.data_len = len(dataset)


    def collate_fn(batch):
        batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)
        if "avec2014" in str(config.data_dir).lower():
            score = [sample[3] for sample in batch]
            two_class_labels = calculate_label(score)
            labels = torch.tensor(two_class_labels)
        if "daic-woz" in str(config.data_dir).lower():
            score = [sample[3] for sample in batch]
            two_class_labels = [sample[4] for sample in batch]
            labels = torch.tensor(two_class_labels)
        if "search" in str(config.data_dir).lower():
            score = [sample[3] for sample in batch]
            two_class_labels = calculate_labels_search(score)
            labels = torch.tensor(two_class_labels)

        visual = pad_sequence([torch.FloatTensor(sample[0]) for sample in batch])
        acoustic = pad_sequence([torch.FloatTensor(sample[1]) for sample in batch])

        return visual, acoustic, labels


    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn)

    return data_loader
