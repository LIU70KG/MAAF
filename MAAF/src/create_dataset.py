import pickle
import numpy as np
from collections import Counter
import torch
from sklearn.preprocessing import StandardScaler


def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def calculate_label(scores):
    labels = []
    for score in scores:
        if score <= 13:
            labels.append(0)  # 正常
        else:
            labels.append(1)  # 抑郁

    return labels


def calculate_labels_search(scores):
    labels = []
    for score in scores:
        if score <= 9:
            labels.append(0)  # 正常
        else:
            labels.append(1)  # 抑郁
    return labels


# SEARCH 正常训练用
class SEARCH:
    def __init__(self, config):

        DATA_PATH = str(config.dataset_dir)

        # If cached data if already exists
        try:
            self.train = load_pickle(DATA_PATH + '/SEARCH_data_train.pkl')
            self.test = load_pickle(DATA_PATH + '/SEARCH_data_test.pkl')

            # usedDatas = []
            # for line in self.train:
            #     if line[3]<=9 or line[3]>=20:
            #         paragraph_inforamtion = [line]
            #         usedDatas.extend(paragraph_inforamtion)
            # self.train = usedDatas  # 2488个，健康+重度抑郁+非常重度抑郁
            #
            # usedDatas = []
            # for line in self.test:
            #     if line[3]<=9 or line[3]>=20:
            #         paragraph_inforamtion = [line]
            #         usedDatas.extend(paragraph_inforamtion)
            # self.test = usedDatas  # 618个，健康+重度抑郁+非常重度抑郁
        except:
            print("N0 SEARCH file")

    def get_shample_number(self, mode):

        if mode == "train":
            score = [sample[3] for sample in self.train]
            two_class_labels = calculate_labels_search(score)
            mut_class_labels = [sample[4] for sample in self.train]
        elif mode == "test":
            score = [sample[3] for sample in self.test]
            two_class_labels = calculate_labels_search(score)
            mut_class_labels = [sample[4] for sample in self.train]
        else:
            print("Mode is not set properly (train/test)")
            exit()

        counter = Counter(two_class_labels)  # Counter({0: 1972, 2: 792, 1: 689, 3: 222, 4: 206})3881
        shample_number = [v for k, v in sorted(counter.items())]
        return shample_number


    def get_data(self, mode):

        if mode == "train":
            return self.train
        elif mode == "test":
            return self.test
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()

def add_awgn_ndarray_by_snr(x, snr_db, eps=1e-12):
    """
    x: ndarray, shape (T, D)，例如 (42, 214)
    snr_db: 目标 SNR(dB)，如 20 / 10 / 5
    """
    # 1. 计算信号功率（均值方差）
    mean = np.mean(x)
    signal_power = np.mean((x - mean) ** 2)

    # 2. 计算目标噪声功率
    noise_power = signal_power / (10 ** (snr_db / 10.0))

    # 3. 生成高斯白噪声 并缩放
    noise = np.random.randn(*x.shape) * np.sqrt(noise_power + eps)

    # 4. 加噪
    return x + noise


# # SEARCH 视频加噪声
# class SEARCH:
#     def __init__(self, config):
#
#         DATA_PATH = str(config.dataset_dir)
#
#         self.train = load_pickle(DATA_PATH + '/SEARCH_data_train.pkl')
#         self.test = load_pickle(DATA_PATH + '/SEARCH_data_test.pkl')
#
#         # usedDatas = []
#         # for line in self.train:
#         #     if line[3]<=9 or line[3]>=20:
#         #         paragraph_inforamtion = [line]
#         #         usedDatas.extend(paragraph_inforamtion)
#         # self.train = usedDatas  # 2488个，健康+重度抑郁+非常重度抑郁
#         #
#         # usedDatas = []
#         # for line in self.test:
#         #     if line[3]<=9 or line[3]>=20:
#         #         paragraph_inforamtion = [line]
#         #         usedDatas.extend(paragraph_inforamtion)
#         # self.test = usedDatas  # 618个，健康+重度抑郁+非常重度抑郁
#
#         #----------视频加噪声
#         usedDatas = []
#         for line in self.train:
#             visual_fea = line[0]
#             x_norm = StandardScaler().fit_transform(visual_fea)  # 归一化
#             audio = StandardScaler().fit_transform(line[1])  # 音频归一化
#             visual_fea = add_awgn_ndarray_by_snr(x_norm, 10) # 设置目标 SNR = 10dB
#             paragraph_inforamtion = [(visual_fea, audio, line[2], line[3], line[4])]
#             usedDatas.extend(paragraph_inforamtion)
#         self.train = usedDatas
#
#         usedDatas = []
#         for line in self.test:
#             visual_fea = line[0]
#             x_norm = StandardScaler().fit_transform(visual_fea)  # 归一化
#             audio = StandardScaler().fit_transform(line[1])  # 音频归一化
#             visual_fea = add_awgn_ndarray_by_snr(x_norm, 10) # 设置目标 SNR = 10dB
#             paragraph_inforamtion = [(visual_fea, audio, line[2], line[3], line[4])]
#             usedDatas.extend(paragraph_inforamtion)
#         self.test = usedDatas
#
#     def get_shample_number(self, mode):
#
#         if mode == "train":
#             score = [sample[3] for sample in self.train]
#             two_class_labels = calculate_labels_search(score)
#             mut_class_labels = [sample[4] for sample in self.train]
#         elif mode == "test":
#             score = [sample[3] for sample in self.test]
#             two_class_labels = calculate_labels_search(score)
#             mut_class_labels = [sample[4] for sample in self.train]
#         else:
#             print("Mode is not set properly (train/test)")
#             exit()
#
#         counter = Counter(two_class_labels)  # Counter({0: 1972, 2: 792, 1: 689, 3: 222, 4: 206})3881
#         shample_number = [v for k, v in sorted(counter.items())]
#         return shample_number
#
#
#     def get_data(self, mode):
#
#         if mode == "train":
#             return self.train
#         elif mode == "test":
#             return self.test
#         else:
#             print("Mode is not set properly (train/dev/test)")
#             exit()
#
#
#
# # SEARCH 音频加噪声
# class SEARCH:
#     def __init__(self, config):
#
#         DATA_PATH = str(config.dataset_dir)
#
#         self.train = load_pickle(DATA_PATH + '/SEARCH_data_train.pkl')
#         self.test = load_pickle(DATA_PATH + '/SEARCH_data_test.pkl')
#
#         # usedDatas = []
#         # for line in self.train:
#         #     if line[3]<=9 or line[3]>=20:
#         #         paragraph_inforamtion = [line]
#         #         usedDatas.extend(paragraph_inforamtion)
#         # self.train = usedDatas  # 2488个，健康+重度抑郁+非常重度抑郁
#         #
#         # usedDatas = []
#         # for line in self.test:
#         #     if line[3]<=9 or line[3]>=20:
#         #         paragraph_inforamtion = [line]
#         #         usedDatas.extend(paragraph_inforamtion)
#         # self.test = usedDatas  # 618个，健康+重度抑郁+非常重度抑郁
#
#
#         #---------- 音频加噪声
#         usedDatas = []
#         for line in self.train:
#             visual = StandardScaler().fit_transform(line[0])  # 视觉归一化
#             audio_fea = line[1]
#             x_norm = StandardScaler().fit_transform(audio_fea)  # 音频归一化
#             audio_fea = add_awgn_ndarray_by_snr(x_norm, 10) # 设置目标 SNR = 10dB
#             paragraph_inforamtion = [(visual, audio_fea, line[2], line[3], line[4])]
#             usedDatas.extend(paragraph_inforamtion)
#         self.train = usedDatas
#
#         usedDatas = []
#         for line in self.test:
#             visual = StandardScaler().fit_transform(line[0])  # 视觉归一化
#             audio_fea = line[1]
#             x_norm = StandardScaler().fit_transform(audio_fea)  # 音频归一化
#             audio_fea = add_awgn_ndarray_by_snr(x_norm, 10) # 设置目标 SNR = 10dB
#             paragraph_inforamtion = [(visual, audio_fea, line[2], line[3], line[4])]
#             usedDatas.extend(paragraph_inforamtion)
#         self.test = usedDatas
#
#     def get_shample_number(self, mode):
#
#         if mode == "train":
#             score = [sample[3] for sample in self.train]
#             two_class_labels = calculate_labels_search(score)
#             mut_class_labels = [sample[4] for sample in self.train]
#         elif mode == "test":
#             score = [sample[3] for sample in self.test]
#             two_class_labels = calculate_labels_search(score)
#             mut_class_labels = [sample[4] for sample in self.train]
#         else:
#             print("Mode is not set properly (train/test)")
#             exit()
#
#         counter = Counter(two_class_labels)  # Counter({0: 1972, 2: 792, 1: 689, 3: 222, 4: 206})3881
#         shample_number = [v for k, v in sorted(counter.items())]
#         return shample_number
#
#
#     def get_data(self, mode):
#
#         if mode == "train":
#             return self.train
#         elif mode == "test":
#             return self.test
#         else:
#             print("Mode is not set properly (train/dev/test)")
#             exit()



class DAIC_WOZ:
    def __init__(self, config):

        DATA_PATH = str(config.dataset_dir)

        # If cached data if already exists
        try:

            self.dev = load_pickle(DATA_PATH + '/valid_data_paragraph_concat.pkl')['valid']  # 35
            self.train = load_pickle(DATA_PATH + '/train_valid_data_paragraph_concat.pkl')['train_valid']  # 142
            self.test = load_pickle(DATA_PATH + '/test_data_paragraph_concat.pkl')['test']  # 47
            usedDatas = []
            for line in self.dev:
                video_in_line = line[0][0]
                audio_in_line = line[0][1]
                label_in_line = line[1]
                number = line[2]
                paragraph_inforamtion = [(video_in_line, audio_in_line, number, label_in_line[1], label_in_line[0])]
                usedDatas.extend(paragraph_inforamtion)
            self.dev = usedDatas

            usedDatas = []
            for line in self.train:
                video_in_line = line[0][0]
                audio_in_line = line[0][1]
                label_in_line = line[1]
                number = line[2]
                paragraph_inforamtion = [(video_in_line, audio_in_line, number, label_in_line[1], label_in_line[0])]
                usedDatas.extend(paragraph_inforamtion)
            self.train = usedDatas

            usedDatas = []
            for line in self.test:
                video_in_line = line[0][0]
                audio_in_line = line[0][1]
                label_in_line = line[1]
                number = line[2]
                paragraph_inforamtion = [(video_in_line, audio_in_line, number, label_in_line[1], label_in_line[0])]
                usedDatas.extend(paragraph_inforamtion)
            self.test = usedDatas
        except:
            print("N0 DAIC_WOZ file")

    def get_data(self, mode):

        if mode == "train":
            return self.train
        elif mode == "dev":
            return self.dev
        elif mode == "test":
            return self.test
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()


    def get_shample_number(self, mode):

        if mode == "train":
            # score = [sample[3] for sample in self.train]
            two_class_labels = [sample[4] for sample in self.train]
        elif mode == "dev":
            # score = [sample[3] for sample in self.dev]
            two_class_labels = [sample[4] for sample in self.train]
        elif mode == "test":
            # score = [sample[3] for sample in self.test]
            two_class_labels = [sample[4] for sample in self.train]
        else:
            print("Mode is not set properly (train/dev/test/train_dev)")
            exit()

        counter = Counter(two_class_labels)
        shample_number = [v for k, v in sorted(counter.items())]
        return shample_number




class AVEC2014:
    def __init__(self, config):
        self.config = config
        DATA_PATH = str(config.dataset_dir)

        # If cached data if already exists
        try:
            self.test = load_pickle(DATA_PATH + '/fea_test.pkl')
            self.train = load_pickle(DATA_PATH + '/fea_train_dev.pkl')
            self.all = load_pickle(DATA_PATH + '/fea_all.pkl')
        except:
            print("N0 AVEC2014 file")

    def calculate_label(self, scores):
        labels = []
        for score in scores:
            if score <= 13:
                labels.append(0)  # 正常
            else:
                labels.append(1)  # 抑郁

        return labels


    def get_shample_number(self, mode):

        if mode == "train":
            score = [sample[3] for sample in self.train]
            mut_class_labels = [sample[4] for sample in self.train]
            two_class_labels = calculate_label(score)
        elif mode == "test":
            score = [sample[3] for sample in self.test]
            mut_class_labels = [sample[4] for sample in self.test]
            two_class_labels = calculate_label(score)
        else:
            print("Mode is not set properly (train/test)")
            exit()

        # mut_class_labels = calculate_labels(score)
        counter = Counter(two_class_labels)
        # counter = Counter(mut_class_labels)  # Counter({0: 33, 1: 19, 3: 18, 4: 13, 2: 12, 5: 5})
        score_all = [sample[3] for sample in self.all]
        mut_class_all = [sample[4] for sample in self.all]
        a = {sample[2]: sample[3] for sample in self.all}
        sorted_a_dict = dict(sorted(a.items(), key=lambda item: item[1]))

        shample_number = [v for k, v in sorted(counter.items())]
        return shample_number


    def get_data(self, mode):

        if mode == "train":
            return self.train
        elif mode == "test":
            return self.test
        else:
            print("Mode is not set properly (train/dev/test)")
            exit()
