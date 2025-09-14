import torch
from torch import nn
import sys
import crossAttention_transformerEncoder
import torch.nn.functional as F
import math


def merged_strategy(
        hidden_states,
        mode="mean"
):
    """'Pooling layer' that combines the outputs of the transformer for each timestep of the input recording
    by applying a mean, sum or max."""
    if mode == "mean":
        outputs = torch.mean(hidden_states, dim=1)
    elif mode == "sum":
        outputs = torch.sum(hidden_states, dim=1)
    elif mode == "max":
        outputs = torch.max(hidden_states, dim=1)[0]
    else:
        raise Exception(
            "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

    return outputs


class DilatedBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding):
        super().__init__()
        self.reduce_dim = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.CausalDilatedConv1d = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.up_dim = nn.Conv1d(out_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.BatchNorm_in_channels = nn.BatchNorm1d(in_channels)
        self.BatchNorm_out_channels = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # 降维
        out = self.reduce_dim(x)
        out = self.BatchNorm_out_channels(out)
        out = self.relu(out)
        # 空洞卷积
        out = self.CausalDilatedConv1d(out)
        out = self.BatchNorm_out_channels(out)
        out = self.relu(out)
        # out = self.dropout(out)
        # 升维
        out = self.up_dim(out)
        out = self.BatchNorm_in_channels(out)
        out = self.relu(out)

        return out + x


class TFML(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3):
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            dilation = 2 ** i
            out_ch = num_channels[i]
            # padding = (kernel_size - 1) * dilation
            padding = ((kernel_size - 1) * dilation) // 2
            layers.append(DilatedBlock(input_size, out_ch, kernel_size, stride=1, dilation=dilation, padding=padding))
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # x: (batch, input_channels, seq_len)
        y = self.network(x)
        # y = y[:, :, -1]  # 取最后一个时间步
        return y


class BVEM(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        # 池化方式作为通道数：2（avg & max）
        # 模态数作为空间维度：3
        self.conv1x1 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1)
        self.fc = nn.Linear(feature_dim, 1)  # 每个模态输出一个分数（标量）

    def forward(self, a_x_A, a_x_Z, a_x_V):
        """
        输入：三个模态特征 [B, T, C]
        输出：模态权重系数 [B, 3]
        """

        def pool(x):
            avg = F.adaptive_avg_pool1d(x.transpose(1, 2), 1)  # [B, C, 1]
            max_ = F.adaptive_max_pool1d(x.transpose(1, 2), 1)  # [B, C, 1]
            return avg, max_

        # 分别处理每个模态
        avg_pooled_A, max_pooled_A = pool(a_x_A)  # [B, C, 1]
        avg_pooled_Z, max_pooled_Z = pool(a_x_Z)
        avg_pooled_V, max_pooled_V = pool(a_x_V)

        avg_pool = torch.cat([
            avg_pooled_A.permute(0, 2, 1).unsqueeze(1),
            avg_pooled_Z.permute(0, 2, 1).unsqueeze(1),
            avg_pooled_V.permute(0, 2, 1).unsqueeze(1)
        ], dim=1)  # [B, 3, 1, C]
        max_pool = torch.cat([
            max_pooled_A.permute(0, 2, 1).unsqueeze(1),
            max_pooled_Z.permute(0, 2, 1).unsqueeze(1),
            max_pooled_V.permute(0, 2, 1).unsqueeze(1)
        ], dim=1)  # [B, 3, 1, C]

        # 变换维度后拼接：avg 和 max 沿“通道”拼接 → [B, 2, 3, C]
        avg_pool = avg_pool.permute(0, 2, 1, 3)  # [B, 1, 3, C]
        max_pool = max_pool.permute(0, 2, 1, 3)  # [B, 1, 3, C]
        pooled = torch.cat([avg_pool, max_pool], dim=1)  # [B, 2, 3, C]

        # 1x1 卷积融合池化方式 → [B, 1, 3, C]
        fused = self.conv1x1(pooled)  # [B, 1, 3, C]
        fused = fused.squeeze(1)      # [B, 3, C]

        # FC 层计算每个模态的分数 → [B, 3]
        scores = self.fc(fused).squeeze(-1)  # [B, 3]

        # Softmax 得到模态权重（每行和为1）→ [B, 3]
        weights = F.softmax(scores, dim=1)

        return weights

class gateInfoToScore(nn.Module):#根据gate信息向量得到最终分数（置信度）
    def __init__(self, num_of_out,dim_nums,dropout):
        super().__init__()
        self.dense = nn.Linear(dim_nums, int(dim_nums/2))
        self.dense_extra = nn.Linear(dim_nums, dim_nums)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(int(dim_nums/2), num_of_out)

        self.softmax = nn.Softmax(dim=1)


    def forward(self, features, **kwargs):
        x = features
        if (True):# 额外加一层
            x = self.dense_extra(x)
            x = torch.tanh(x)
            x = self.dropout(x)
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        x = self.softmax(x)

        return x


class ClassificationHead(nn.Module):

    def __init__(self, num_labels,dim_nums,dropout):
        super().__init__()
        self.dense = nn.Linear(dim_nums, int(dim_nums/4))
        self.dense_extra = nn.Linear(dim_nums, dim_nums)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(int(dim_nums/4), num_labels)

    def forward(self, features, **kwargs):
        x = features
        if (False):
            x = self.dense_extra(x)  # 额外加一层
            x = torch.tanh(x)  # 额外加一层
            x = self.dropout(x)  # 额外加一层
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.relu(x)
        # x = torch.tanh(x)
        x = self.dropout(x)



        x = self.out_proj(x)
        return x


class AZVmodel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.needAudioLSTM = config.needChangeAudioDimInLstm
        self.lstm_ForA = nn.LSTM(input_size=config.acoustic_size,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.lstm = nn.LSTM(input_size=config.visual_size,
                            hidden_size=128,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=False)

        self.tfml = TFML(input_size=128, num_channels=[64]*4, kernel_size=3)
        self.bvem = BVEM(feature_dim=128)
        dim = 128
        head = 8

        self.fml_A_1 = nn.TransformerEncoderLayer(d_model=dim, nhead=head)
        self.fml_A_2 = nn.TransformerEncoderLayer(d_model=dim, nhead=head)
        self.fml_V_1 = nn.TransformerEncoderLayer(d_model=dim, nhead=head)
        self.fml_V_2 = nn.TransformerEncoderLayer(d_model=dim, nhead=head)

        ffn_hidden = dim * 4
        self.fusion_A_V = crossAttention_transformerEncoder.EncoderLayer(d_model=dim, ffn_hidden=ffn_hidden,
                                                                         n_head=head, drop_prob=0.1)
        self.fusion_V_A = crossAttention_transformerEncoder.EncoderLayer(d_model=dim, ffn_hidden=ffn_hidden,
                                                                        n_head=head, drop_prob=0.1)

        # self.d_fusion_Z_A = crossAttention_transformerEncoder.EncoderLayer(d_model=dim, ffn_hidden=ffn_hidden,
        #                                                                  n_head=head, drop_prob=0.1)
        # self.d_fusion_Z_V = crossAttention_transformerEncoder.EncoderLayer(d_model=dim, ffn_hidden=ffn_hidden,
        #                                                                    n_head=head, drop_prob=0.1)
        self.d_fusion_Z = nn.TransformerEncoderLayer(d_model=dim, nhead=head)


        self.gateScoreGetter = gateInfoToScore(num_of_out=3,dim_nums=3*dim,dropout=0)

        self.classifier_A = ClassificationHead(num_labels=2, dim_nums=dim, dropout=0)
        self.classifier_Z = ClassificationHead(num_labels=2, dim_nums=dim, dropout=0)
        self.classifier_V = ClassificationHead(num_labels=2, dim_nums=dim, dropout=0)

        self.softmax = nn.Softmax(dim=-1)

        # self.lmf_fusion = LMF(input_dims=[128, 128], rank=4, output_dim=128, dropout=0.2)
        # LMF
        self.factor_A = nn.Parameter(torch.Tensor(4, 128, 128))
        self.factor_V = nn.Parameter(torch.Tensor(4, 128, 128))
        nn.init.xavier_uniform_(self.factor_A)
        nn.init.xavier_uniform_(self.factor_V)

    def forward(self,x_A,x_V):
        # self.lstm_ForA要求输入shape = [batch_size, sequence_length, input_size]= [B, T, config.acoustic_size]
        x_A = x_A.permute(1, 0, 2)   # [sequence_length, batch_size, input_size] -> [batch_size, sequence_length, input_size]
        x_V = x_V.permute(1, 0, 2)
        if self.needAudioLSTM:
            x_A,_ = self.lstm_ForA(x_A)
        x_V , _ = self.lstm(x_V)
        # TFML 输入要求是(batch_size,, d_model, seq_len)
        x_A = x_A.permute(0, 2, 1)  # shape: [64, 20, 128] -> [64, 128, 20]
        x_V = x_V.permute(0, 2, 1)
        x_A = self.tfml(x_A)
        x_V = self.tfml(x_V)  # shape: [64, 20, 128]

        # TransformerEncoderLayer的self-attention+FFN
        x_A = x_A.permute(2, 0, 1)  # shape: [64, 128, 20] ->  [20, 64, 128]
        x_V = x_V.permute(2, 0, 1)
        x_A_f1 = self.fml_A_1(x_A) # self.fml_A_1的输入要求是(seq_len, batch_size, d_model)
        x_V_f1 = self.fml_V_1(x_V)

        # Fusion MuIT
        # x_Z = self.fusion_A_V(x_A,x_V)+self.fusion_V_A(x_V,x_A)
        # x_Z = x_Z.permute(1, 0, 2)

        # Fusion  TFN
        # a = x_A[-1, :, :]  # 取最后一个时间步
        # v = x_V[-1, :, :]  # 取最后一个时间步
        # x_Z = torch.bmm(a.unsqueeze(2), v.unsqueeze(1))
        # x_Z = torch.bmm(x_A[:, -1:].transpose(2, 1), x_V[:, -1:])  # 是不是这个将可以了

        # Fusion  LMF
        a = x_A[-1, :, :]  # [B, 128]
        v = x_V[-1, :, :]  # [B, 128]
        A_proj = torch.einsum('bd,rdm->brm', a, self.factor_A)
        V_proj = torch.einsum('bd,rdm->brm', v, self.factor_V)
        # Element-wise multiply across modalities (bilinear fusion)
        x_Z = A_proj * V_proj

        # 改成[B, T, C]
        x_A_f1 = x_A_f1.permute(1, 0, 2)  # shape: [20, 64, 128] ->  [64, 20, 128]
        x_V_f1 = x_V_f1.permute(1, 0, 2)
        gateScore = self.bvem(x_A_f1, x_Z, x_V_f1)  # shape: [B, 3]

        a_x_A = merged_strategy(x_A_f1, mode="mean")
        a_x_Z = merged_strategy(x_Z, mode="mean")
        a_x_V = merged_strategy(x_V_f1, mode="mean")
        out_A = self.classifier_A(a_x_A)
        out_Z = self.classifier_Z(a_x_Z)
        out_V = self.classifier_V(a_x_V)

        #这步的softmax后是正常的输出，但是为了计算loss，后面输出前进行了log，因为【softmax + torch.log + nn.NLLLoss = nn.CrossEntropyLoss】
        out_A = self.softmax(out_A)
        out_Z = self.softmax(out_Z)
        out_V = self.softmax(out_V)

        out_concated = torch.stack((out_A,out_Z,out_V),dim = 1)
        gateScore_us = gateScore.unsqueeze(dim=-1)
        out = gateScore_us * out_concated
        out = merged_strategy(out, mode="sum")

        out_A = torch.log(out_A)
        out_Z = torch.log(out_Z)
        out_V = torch.log(out_V)
        out = torch.log(out)
        # 注意，由于nn.CrossEntropyLoss里面包含了softmax，而我们在这里手动进行了softmax，所以必须用nn.NLLLoss，【softmax + torch.log + nn.NLLLoss = nn.CrossEntropyLoss】
        return out, out_A,out_Z,out_V,gateScore


class Simple_concat(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_dims = 128
        self.rank = 4
        self.output_dim = 32
        self.dropout = nn.Dropout(0.4)


        self.needAudioLSTM = config.needChangeAudioDimInLstm
        self.lstm_ForA = nn.LSTM(input_size=config.acoustic_size,
                                 hidden_size=self.input_dims,
                                 num_layers=2,
                                 batch_first=True,
                                 bidirectional=False)

        self.lstm = nn.LSTM(input_size=config.visual_size,
                            hidden_size=self.input_dims,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)


        self.classifier = nn.Sequential(
            nn.Linear(self.input_dims*2, self.output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.output_dim, 2)
        )


    def forward(self, x_A, x_V):
        # self.lstm_ForA要求输入shape = [batch_size, sequence_length, input_size]= [B, T, config.acoustic_size]
        x_A = x_A.permute(1, 0, 2)  # [sequence_length, batch_size, input_size] -> [batch_size, sequence_length, input_size]
        x_V = x_V.permute(1, 0, 2)
        output, (h_n, c_n) = self.lstm_ForA(x_A)
        x_A = h_n[-1]
        output, (h_n, c_n) = self.lstm(x_V)
        x_V = h_n[-1]

        fusion = torch.cat([x_A, x_V], dim=1)  # shape: [B, 256]
        # fusion = self.dropout(fusion)
        logits = self.classifier(fusion)  # [B, num_labels]
        return logits


class LMF(nn.Module):

    def __init__(self, config):
        """
        input_dims: list of input feature dims, e.g. [128, 128] for two modalities
        rank: low-rank approximation factor (typically 4~10)
        output_dim: final fused output dimension
        """
        super().__init__()
        self.input_dims = 128
        self.rank = 4
        self.output_dim = 64
        self.dropout = nn.Dropout(0.4)

        # Define factor matrices for each modality
        self.factor_A = nn.Parameter(torch.Tensor(self.rank, self.input_dims, self.output_dim))
        self.factor_V = nn.Parameter(torch.Tensor(self.rank, self.input_dims, self.output_dim))

        # Fusion bias
        self.fusion_bias = nn.Parameter(torch.Tensor(self.output_dim))

        self.needAudioLSTM = config.needChangeAudioDimInLstm
        self.lstm_ForA = nn.LSTM(input_size=config.acoustic_size,
                            hidden_size=self.input_dims,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        self.lstm = nn.LSTM(input_size=config.visual_size,
                            hidden_size=self.input_dims,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        self.classifier = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.output_dim // 2, 2)
        )

        # Init
        nn.init.xavier_uniform_(self.factor_A)
        nn.init.xavier_uniform_(self.factor_V)
        nn.init.constant_(self.fusion_bias, 0)

    def forward(self, x_A, x_V):
        """
        x_A: [B, D_A]  e.g. [64, 128]
        x_V: [B, D_V]  e.g. [64, 128]
        return: [B, output_dim]
        """
        # self.lstm_ForA要求输入shape = [batch_size, sequence_length, input_size]= [B, T, config.acoustic_size]
        x_A = x_A.permute(1, 0, 2)   # [sequence_length, batch_size, input_size] -> [batch_size, sequence_length, input_size]
        x_V = x_V.permute(1, 0, 2)
        output, (h_n, c_n) = self.lstm_ForA(x_A)
        x_A = h_n[-1]
        output, (h_n, c_n) = self.lstm(x_V)
        x_V = h_n[-1]
        # [B, r, d_out]
        A_proj = torch.einsum('bd,rdm->brm', x_A, self.factor_A)
        V_proj = torch.einsum('bd,rdm->brm', x_V, self.factor_V)

        # Element-wise multiply across modalities (bilinear fusion)
        fusion = A_proj * V_proj  # [B, r, d_out]
        fusion = torch.sum(fusion, dim=1) + self.fusion_bias  # [B, d_out]
        # fusion = self.dropout(fusion)
        logits = self.classifier(fusion)  # [B, num_labels]
        return logits


class TFN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.input_dims = 64
        self.dropout = nn.Dropout(0.4)

        self.needAudioLSTM = config.needChangeAudioDimInLstm
        self.lstm_ForA = nn.LSTM(input_size=config.acoustic_size,
                            hidden_size=self.input_dims,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        self.lstm = nn.LSTM(input_size=config.visual_size,
                            hidden_size=self.input_dims,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        self.classifier = nn.Sequential(
            nn.Linear((self.input_dims+1)**2, self.input_dims),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.input_dims, 2)
        )


    def forward(self, x_A, x_V):
        # self.lstm_ForA要求输入shape = [batch_size, sequence_length, input_size]= [B, T, config.acoustic_size]
        x_A = x_A.permute(1, 0, 2)   # [sequence_length, batch_size, input_size] -> [batch_size, sequence_length, input_size]
        x_V = x_V.permute(1, 0, 2)
        output, (h_n, c_n) = self.lstm_ForA(x_A)
        a = h_n[-1]
        # a = output[:, -1, :]  # [B, hidden_size]
        output, (h_n, c_n) = self.lstm(x_V)
        v = h_n[-1]
        # v = output[:, -1, :]  # [B, hidden_size]

        a = torch.cat([a, torch.ones(a.size(0), 1).to(a.device)], dim=1)
        v = torch.cat([v, torch.ones(v.size(0), 1).to(v.device)], dim=1)
        fusion = torch.bmm(a.unsqueeze(2), v.unsqueeze(1)).view(a.size(0), -1)
        # fusion = self.dropout(fusion)
        logits = self.classifier(fusion)  # [B, num_labels]
        return logits




class SimpleMulT(nn.Module):
    def __init__(self, config):
        super().__init__()
        super().__init__()

        self.needAudioLSTM = config.needChangeAudioDimInLstm
        self.lstm_ForA = nn.LSTM(input_size=config.acoustic_size,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        self.lstm = nn.LSTM(input_size=config.visual_size,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=False)

        dim = 128
        head = 8
        ffn_hidden = dim * 4
        self.fusion_A_V = crossAttention_transformerEncoder.EncoderLayer(d_model=dim, ffn_hidden=ffn_hidden,
                                                                         n_head=head, drop_prob=0.1)
        self.fusion_V_A = crossAttention_transformerEncoder.EncoderLayer(d_model=dim, ffn_hidden=ffn_hidden,
                                                                        n_head=head, drop_prob=0.1)
        self.classifier_Z = ClassificationHead(num_labels=2, dim_nums=dim, dropout=0)


    def forward(self, x_A, x_V):
        # self.lstm_ForA要求输入shape = [batch_size, sequence_length, input_size]= [B, T, config.acoustic_size]
        x_A = x_A.permute(1, 0, 2)   # [sequence_length, batch_size, input_size] -> [batch_size, sequence_length, input_size]
        x_V = x_V.permute(1, 0, 2)
        if self.needAudioLSTM:
            x_A,_ = self.lstm_ForA(x_A)
        x_V , _ = self.lstm(x_V)

        x_Z = self.fusion_A_V(x_A,x_V)+self.fusion_V_A(x_V,x_A)
        a_x_Z = merged_strategy(x_Z, mode="mean")
        out_Z = self.classifier_Z(a_x_Z)
        return out_Z  # 融合后的表示 [B, T, d_model]


