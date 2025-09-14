import os
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch
import torch.nn as nn
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
from utils import to_gpu
import models
from shutil import copyfile, rmtree
from sklearn.metrics import precision_score, recall_score, f1_score

class Solver(object):
    def __init__(self, train_config, test_config, train_data_loader, test_data_loader, is_train=True, model=None):

        self.train_config = train_config
        self.epoch_i = 0
        self.train_data_loader = train_data_loader
        self.test_data_loader = test_data_loader
        self.is_train = is_train
        self.model = model
    
    # @time_desc_decorator('Build Graph')
    def build(self, cuda=True):

        if self.model is None:
            self.model = getattr(models, self.train_config.model)(self.train_config)
        
        # Final list
        for name, param in self.model.named_parameters():
            if 'weight_hh' in name:
                nn.init.orthogonal_(param)
            # print('\t' + name, param.requires_grad)
        
        if torch.cuda.is_available() and cuda:
            self.model.cuda()

        if self.is_train:
            self.optimizer = self.train_config.optimizer(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.train_config.learning_rate)  # , weight_decay=1e-3


    # @time_desc_decorator('Training Start!')
    def train(self):
        curr_patience = patience = self.train_config.patience
        num_trials = 1
        # self.criterion = criterion = nn.MSELoss(reduction="mean")
        # self.MAE = nn.L1Loss(reduction='mean')

        loss_CE = nn.CrossEntropyLoss(weight=self.train_config.weights.cuda(), reduction="mean")
        loss = nn.NLLLoss(weight=self.train_config.weights.cuda(), reduction="mean")
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.5)

        # self.tsne(mode="test", best=True)
        # self.picture(mode="test", best=True)

        continue_epochs = 0
        # if os.path.isfile('checkpoints/model_2024-07-18_20:15:36.std'):
        #     print("Loading pretrained weights...")
        #     self.model.load_state_dict(torch.load(
        #         f'checkpoints/model_2024-07-18_20:15:36.std'))
        #
        #     self.optimizer.load_state_dict(torch.load(
        #         f'checkpoints/optim_2024-07-18_20:15:36.std'))
        #     continue_epochs = 9
        # print("continue iter:", continue_epochs)

        checkpoints = './checkpoints'
        if os.path.exists(checkpoints):
                rmtree(checkpoints, ignore_errors=False, onerror=None)
        os.makedirs(checkpoints)

        # 用来保存每个epoch的Loss和acc以便最后画图
        train_losses = []
        train_acces = []
        test_losses = []
        test_acces = []
        best_acc = 0.0
        best_F1 = 0
        precision_bestF1, recall_bestF1, f1_bestF1, accuracy_bestF1 = 0, 0, 0, 0
        for e in range(continue_epochs, self.train_config.n_epoch):
            print(f"-----------------------------------epoch{e}---------------------------------------")
            print(f"//Current patience: {curr_patience}, current trial: {num_trials}.//")
            self.model.train()
            train_loss_all = 0
            target_loss_all, oneMoLoss_all, Loss_score_all = 0, 0, 0
            y_true, y_pred = [], []
            for batch in self.train_data_loader:
                # self.model.zero_grad()
                self.optimizer.zero_grad()
                video, audio, targetLabel = batch
                video = to_gpu(video)
                audio = to_gpu(audio)
                targetLabel = to_gpu(targetLabel)
                if self.train_config.model=="AZVmodel" or self.train_config.model=="AZVmodel_TFN":  # AZV模型比较特别
                    output, output_A, output_Z, output_V, output_gateScore = self.model(audio, video)
                else:
                    output = self.model(audio, video)

                # ******************************************loss

                if self.train_config.model=="AZVmodel" or self.train_config.model=="AZVmodel_TFN":  # AZV模型比较特别
                    score_label = []
                    for i in range(len(output)):
                        cross_A = loss(output_A[i], targetLabel[i]).item()
                        cross_Z = loss(output_Z[i], targetLabel[i]).item()
                        cross_V = loss(output_V[i], targetLabel[i]).item()

                        cross = np.array([cross_A, cross_Z, cross_V])
                        minCrossMo = np.argmin(cross)  # 交叉熵最小的那个模态，0=A,1=Z,2=V
                        score_label.append(minCrossMo)
                    score_label = torch.LongTensor(score_label)
                    score_label = to_gpu(score_label)

                    gateScore_log = torch.log(output_gateScore)
                    lossFuc_forScore = nn.NLLLoss()
                    Loss_score = lossFuc_forScore(gateScore_log, score_label)

                    Loss_A = loss(output_A, targetLabel)
                    Loss_Z = loss(output_Z, targetLabel)
                    Loss_V = loss(output_V, targetLabel)
                    oneMoLoss = (Loss_A + Loss_Z + Loss_V)
                    target_loss = loss(output, targetLabel)
                    # Loss = target_loss
                    # Loss = Loss_Z
                    # Loss = target_loss + self.train_config.oneMoLossRatio * oneMoLoss
                    Loss = target_loss + self.train_config.oneMoLossRatio * oneMoLoss + self.train_config.scoreLossRatio * Loss_score
                    target_loss_all = target_loss_all + target_loss
                    oneMoLoss_all = oneMoLoss_all + oneMoLoss
                    Loss_score_all = Loss_score_all + Loss_score
                else:
                    Loss = loss_CE(output, targetLabel)

                self.optimizer.zero_grad()
                Loss.backward()
                torch.nn.utils.clip_grad_norm_([param for param in self.model.parameters() if param.requires_grad],
                                               self.train_config.clip)
                self.optimizer.step()

                train_loss_all = train_loss_all + Loss
                _, pred = output.max(1)  # 分类任务
                y_pred.append(pred.detach().cpu().numpy())
                y_true.append(targetLabel.detach().cpu().numpy())

            y_true = np.concatenate(y_true, axis=0).squeeze()
            y_pred = np.concatenate(y_pred, axis=0).squeeze()

            num_correct = (y_pred == y_true).sum().item()
            trainAcc_epoch=num_correct/len(y_pred)
            precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
            f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

            sumLoss_epoch = train_loss_all / len(self.train_data_loader)
            target_loss_epoch = target_loss_all / len(self.train_data_loader)
            oneMoLoss_epoch = oneMoLoss_all / len(self.train_data_loader)
            Loss_score_epoch = Loss_score_all / len(self.train_data_loader)
            # train_acces.append(trainAcc_epoch)
            # train_losses.append(trainLoss_epoch)

            print("train---sumLoss: {}, target_loss: {}, oneMoLoss: {}, Loss_score: {}".format(sumLoss_epoch, target_loss_epoch, oneMoLoss_epoch, Loss_score_epoch))
            print(" precision_macro: {}, recall_macro: {}, f1_macro: {}, Acc: {} ".format(precision_macro, recall_macro, f1_macro, trainAcc_epoch))


            # 测试步骤开始
            self.model.eval()
            eval_loss = 0
            eval_acc = 0
            y_true, y_pred = [], []

            with torch.no_grad():
                for batch in self.test_data_loader:
                    self.optimizer.zero_grad()
                    video, audio, targetLabel = batch
                    video = to_gpu(video)
                    audio = to_gpu(audio)
                    targetLabel = to_gpu(targetLabel)

                    if self.train_config.model == "AZVmodel" or self.train_config.model=="AZVmodel_TFN":  # AZV模型比较特别
                        output, output_A, output_Z, output_V, output_gateScore = self.model(audio, video)
                    else:
                        output = self.model(audio, video)

                    if self.train_config.model == "AZVmodel" or self.train_config.model == "AZVmodel_TFN":  # AZV模型比较特别
                        Loss = loss(output, targetLabel)
                    else:
                        Loss = loss_CE(output, targetLabel)

                    eval_loss += Loss

                    _, pred = output.max(1)  # 分类任务
                    y_pred.append(pred.detach().cpu().numpy())
                    y_true.append(targetLabel.detach().cpu().numpy())

                y_true = np.concatenate(y_true, axis=0).squeeze()
                y_pred = np.concatenate(y_pred, axis=0).squeeze()

                num_correct = (y_pred == y_true).sum().item()
                evalAcc_epoch=num_correct/len(y_pred)
                precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
                recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
                f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)

                evalLoss_epoch = eval_loss / len(self.test_data_loader)
                # test_acces.append(evalAcc_epoch)
                # test_losses.append(evalLoss_epoch)

                print("test---target_loss: {}, Acc: {}, precision_macro: {}, recall_macro: {}, f1_macro: {}".format(evalLoss_epoch, evalAcc_epoch, precision_macro, recall_macro, f1_macro))

            # test_mae_history.append(mae)
            if f1_macro > best_F1:
                best_F1 = f1_macro
                precision_bestF1, recall_bestF1, f1_bestF1, accuracy_bestF1 = precision_macro, recall_macro, f1_macro, evalAcc_epoch
                flag = 1
                print("------------------Found new best model on test set!----------------")
                print(f"epoch: {e}")
                print("precision: ", precision_macro)
                print("recall: ", recall_macro)
                print("F1: ", f1_macro)
                print("accuracy: ", evalAcc_epoch)

                if not os.path.exists('checkpoints'): os.makedirs('checkpoints')
                torch.save(self.model.state_dict(), f'checkpoints/model_{self.train_config.name}.std')
                torch.save(self.optimizer.state_dict(), f'checkpoints/optim_{self.train_config.name}.std')
                curr_patience = patience
            else:
                curr_patience -= 1
                if curr_patience <= -1:
                    print("Running out of patience, loading previous best model.")
                    num_trials -= 1
                    curr_patience = patience
                    self.model.load_state_dict(torch.load(f'checkpoints/model_{self.train_config.name}.std'))
                    self.optimizer.load_state_dict(torch.load(f'checkpoints/optim_{self.train_config.name}.std'))
                    lr_scheduler.step()
                    # print(f"Current learning rate: {self.optimizer.state_dict()['param_groups'][0]['lr']}")
            
            if num_trials <= 0:
                print("Running out of patience, early stopping.")
                break


        print("------------------best F1 on test set----------------")
        print('_precision:%.4f./ recall:%.4f./ f1:%.4f./ accuracy:%.4f./' % (precision_bestF1, recall_bestF1, f1_bestF1, accuracy_bestF1))

        # 判断文件是否存在
        if not os.path.exists(self.train_config.best_model_Configuration_Log):
            # 如果文件不存在，则创建文件
            with open(self.train_config.best_model_Configuration_Log, 'w') as f:
                pass  # 创建一个空文件

        with open(self.train_config.best_model_Configuration_Log, 'a', encoding="utf-8") as F1:
            line = 'oneMoLossRatio:{oneMoLossRatio} | scoreLossRatio:{scoreLossRatio}\n ' \
                   'test_best_f1:-----------{f1}------------ | precision:{precision} | recall:{recall} | accuracy:{accuracy}\n' \
                .format(oneMoLossRatio=self.train_config.oneMoLossRatio,
                        scoreLossRatio=self.train_config.scoreLossRatio,
                        precision=precision_bestF1,
                        recall=recall_bestF1,
                        f1=f1_bestF1,
                        accuracy=accuracy_bestF1,
                        )

            print('result saved～')
            F1.write(line)

        checkpoints = 'checkpoints'
        if os.path.exists(checkpoints):
                rmtree(checkpoints, ignore_errors=False, onerror=None)
        os.makedirs(checkpoints)

        return best_F1
        # self.tsne(mode="test", best=True)
        # self.picture(mode="test", best=True)


    def tsne(self, mode=None, to_print=False, best=False):
        assert (mode is not None)
        self.model.eval()

        Feature, Label_area = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

        if best:
            # self.train_config.name = '2024-07-03_22:32:17'  # 文件5
            self.train_config.name = '2024-07-03_22:32:17'
            self.model.load_state_dict(torch.load(
                f'checkpoints/model_{self.train_config.name}.std'))

        with torch.no_grad():
            for batch in self.train_data_loader:
                self.model.zero_grad()
                v, a, y, label_area, label_shifting, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                label_area = to_gpu(label_area)
                label_shifting = to_gpu(label_shifting)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                feature = self.model(v, a, l, bert_sent, bert_sent_type, bert_sent_mask,
                                                       label_area, mode="tsne")


                Feature.append(feature.detach().cpu().numpy())
                Label_area.append(label_area.detach().cpu().numpy())

            for batch in dataloader:
                self.model.zero_grad()
                v, a, y, label_area, label_shifting, l, bert_sent, bert_sent_type, bert_sent_mask = batch

                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                label_area = to_gpu(label_area)
                label_shifting = to_gpu(label_shifting)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                feature = self.model(v, a, l, bert_sent, bert_sent_type, bert_sent_mask,
                                                       label_area, mode="tsne")


                Feature.append(feature.detach().cpu().numpy())
                Label_area.append(label_area.detach().cpu().numpy())

        from sklearn.manifold import TSNE
        import pandas as pd
        import matplotlib.pyplot as plt
        expression = ['healthy', 'light', 'moderate', 'Moderate to severe', 'severe']
        colors = ['green', 'blue', 'yellow', 'orange', 'red']
        features = np.concatenate(Feature, axis=0)
        labels = np.concatenate(Label_area, axis=0)
        tsne1 = TSNE(n_components=2, init="pca", random_state=1, perplexity=5)
        x_tsne1 = tsne1.fit_transform(features)
        print(
            "Data has the {} before tSNE and the following after tSNE {}".format(features.shape[-1], x_tsne1.shape[-1]))
        x_min, x_max = x_tsne1.min(0), x_tsne1.max(0)
        X_norm1 = (x_tsne1 - x_min) / (x_max - x_min)

        ''' plot results of tSNE '''
        fake_df1 = pd.DataFrame(X_norm1, columns=['X', 'Y'])
        fake_df1['Group'] = labels

        group_codes1 = {k: colors[idx] for idx, k in enumerate(np.sort(fake_df1.Group.unique()))}
        fake_df1['colors'] = fake_df1['Group'].apply(lambda x: group_codes1[x])
        mode = ['train']*Feature[0].shape[0] + ['test']*Feature[1].shape[0]
        fake_df1['mode'] = mode

        # 将像素值转换为英寸
        width_in_inches = 1196 / 100
        height_in_inches = 802 / 100
        # 创建固定大小的图像
        fig, ax = plt.subplots(figsize=(width_in_inches, height_in_inches), dpi=100)

        for i in range(fake_df1.Group.unique().shape[0]):
            ax.scatter(X_norm1[fake_df1['Group'] == i, 0],
                       X_norm1[fake_df1['Group'] == i, 1],
                       c=group_codes1[i], label=expression[i], s=70, marker='o', linewidths=1)
        # plt.title('Decomposed features', fontsize=15)
        ax.legend()

        # for i in range(fake_df1.Group.unique().shape[0]):
        #     ax.scatter(X_norm1[(fake_df1['Group'] == i) & (fake_df1['mode'] == 'train'), 0],
        #                X_norm1[(fake_df1['Group'] == i) & (fake_df1['mode'] == 'train'), 1],
        #                c=group_codes1[i], label=expression[i], s=40, marker='o', linewidths=1)
        # # plt.title('Decomposed features', fontsize=15)
        #
        # for i in range(fake_df1.Group.unique().shape[0]):
        #     ax.scatter(X_norm1[(fake_df1['Group'] == i) & (fake_df1['mode'] == 'test'), 0],
        #                X_norm1[(fake_df1['Group'] == i) & (fake_df1['mode'] == 'test'), 1],
        #                c=group_codes1[i], label=expression[i], s=80, marker='*', linewidths=1)
        # # plt.title('Decomposed features', fontsize=15)
        # ax.legend()

        #		plt.legend(loc = 1, fontsize = 'small')
        # ax.legend(fontsize=20, bbox_to_anchor=(-0.015, 0.98, 0.1, 0.1), loc='lower left', ncol=3, columnspacing=1)
        plt.savefig('./figure/TSNE-CMDC5-5-da.png', bbox_inches='tight')
        plt.close("all")


    def picture(self, mode=None, to_print=False, best=False):
        assert (mode is not None)
        self.model.eval()

        Feature, Label_area = [], []

        if mode == "dev":
            dataloader = self.dev_data_loader
        elif mode == "test":
            dataloader = self.test_data_loader

        if best:
            self.train_config.name = '2024-07-04_15:10:57'  # 文件5 -all
            # self.train_config.name = '2024-07-03_22:32:17'  # 文件5 -nocenter
            self.model.load_state_dict(torch.load(
                f'checkpoints/model_{self.train_config.name}.std'))

        with torch.no_grad():
            for batch in self.train_data_loader:
                self.model.zero_grad()
                t, v, a, y, label_area, label_shifting, l = batch

                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                label_area = to_gpu(label_area)
                label_shifting = to_gpu(label_shifting)
                l = to_gpu(l)

                feature = self.model(t, v, a, l, label_area, mode="tsne")

                Feature.append(feature.detach().cpu().numpy())
                Label_area.append(label_area.detach().cpu().numpy())

            for batch in dataloader:
                self.model.zero_grad()
                t, v, a, y, label_area, label_shifting, l = batch

                t = to_gpu(t)
                v = to_gpu(v)
                a = to_gpu(a)
                y = to_gpu(y)
                label_area = to_gpu(label_area)
                label_shifting = to_gpu(label_shifting)
                l = to_gpu(l)
                bert_sent = to_gpu(bert_sent)
                bert_sent_type = to_gpu(bert_sent_type)
                bert_sent_mask = to_gpu(bert_sent_mask)

                feature = self.model(t, v, a, l, bert_sent, bert_sent_type, bert_sent_mask,
                                     label_area, mode="tsne")

                Feature.append(feature.detach().cpu().numpy())
                Label_area.append(label_area.detach().cpu().numpy())

        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt
        expression = ['healthy', 'light', 'moderate', 'Moderate to severe', 'severe']
        colors = ['green', 'blue', 'yellow', 'orange', 'red']
        features = np.concatenate(Feature, axis=0)
        labels = np.concatenate(Label_area, axis=0)
        # 将标签和特征组合在一起，便于排序
        combined = np.hstack((labels, features))
        # 根据标签（即第一列）进行排序
        sorted_combined = combined[combined[:, 0].argsort()]
        # 分离出排序后的标签和特征
        sorted_labels = sorted_combined[:, 0].reshape(-1, 1)
        sorted_features = sorted_combined[:, 1:]
        # 交换sorted_features的两个维度
        transposed_features = sorted_features.T
        # transposed_features = 1 - transposed_features
        # 创建子图
        fig, axes = plt.subplots(1, 1, figsize=(8, 6))

        # 绘制热力图
        sns.heatmap(transposed_features, ax=axes, cmap='viridis', cbar=True)
        # axes.set_title('(a) CMDC5-nocenter')
        axes.set_title('(b) CMDC5')
        axes.set_xlabel('sample')
        axes.set_ylabel('Feature components')
        # axes.add_patch(plt.Rectangle((4, 20), 12, 60, fill=False, edgecolor='red', lw=2))
        # 将sorted_labels转换为适合显示的格式
        sorted_labels_list = [str(int(label[0])) for label in sorted_labels]
        # 设置x轴标签
        axes.set_xticks(np.arange(len(sorted_labels)))
        axes.set_xticklabels(sorted_labels_list, rotation=90)

        # 调整布局
        plt.tight_layout()
        # plt.show()
        plt.savefig('./figure/picture-CMDC5.png', bbox_inches='tight')
        plt.close("all")