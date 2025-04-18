
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, feature_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(feature_dim, feature_dim)
        self.key = nn.Linear(feature_dim, feature_dim)
        self.value = nn.Linear(feature_dim, feature_dim)

    def forward(self, x):
        batch_size, channel_size, seq_len, feature_dim = x.size()
        x = x.view(batch_size, channel_size * seq_len, feature_dim)
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / (feature_dim ** 0.5)
        attention_scores = F.softmax(attention_scores, dim=-1)
        weighted_values = torch.matmul(attention_scores, values)
        weighted_values = weighted_values.view(batch_size, channel_size, seq_len, feature_dim)
        return weighted_values

class ModelAffinity(nn.Module):
    def __init__(
        self,
        bs,
        use_cuda,
    ):
        super(ModelAffinity, self).__init__()
        self.use_cuda = use_cuda
        self.bs = bs
        self.self_attention = SelfAttention(768)
        self.layernorm = nn.LayerNorm(normalized_shape=(4, 768, 768))

        # bert
        self.conv0 = nn.Conv2d(4, 1, kernel_size=1)
        self.conv1 = nn.Conv1d(768, 384, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(384, 1, kernel_size=7, padding=3)
        self.conv3 = nn.Conv1d(768, 384, kernel_size=5, padding=2)
        self.conv4 = nn.Conv1d(384, 1, kernel_size=3, padding=1)
        self.linear = nn.Linear(768, 1)
        self.linear2 = nn.Linear(4, 1)


        # aaindex
        self.fusion1 = FusionModel(20*256, 256)
        self.fusion2 = FusionModel(20*256, 256)
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(768+512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 1)
        self.aa_conv1 = nn.Conv1d(256*3, 256, kernel_size=7, padding=3)
        self.aa_conv2 = nn.Conv1d(256, 1, kernel_size=7, padding=3)
        self.aa_fc = nn.Linear(103, 1)
        self.relu = nn.ReLU()
        self.activation = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.batchnorm = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(768)

    def feature_fusion(self, lchain, hchain, antigen):
        antigen = antigen.unsqueeze(1).unsqueeze(2)  # 32*1*1*768
        lchain = lchain.unsqueeze(1).unsqueeze(3)  # 32*1*768*1
        hchain = hchain.unsqueeze(1).unsqueeze(3)  # 32*1*768*1

        l_dif = torch.abs(lchain - antigen)
        l_mul = lchain * antigen
        l_cat = torch.cat([l_dif, l_mul], 1)

        h_dif = torch.abs(hchain - antigen)
        h_mul = hchain * antigen
        h_cat = torch.cat([h_dif, h_mul], 1)

        ab_cat = torch.cat([l_cat, h_cat], 1)  # 4*768*768
        C = self.layernorm(ab_cat)
        return C

    # module 1: predict from embedding (proteinBert features)
    def map_predict(self, lchain, hchain, antigen):
        if self.use_cuda:
            lchain = lchain
            hchain = hchain
            antigen = antigen

        C = self.feature_fusion(lchain, hchain, antigen)  # [bs, 4, 768, 768]
        B = self.conv0(C)  # torch.Size([bs, 1, 768, 768])
        B = B.squeeze(1)  # torch.Size([bs, 768, 768])
        B = self.relu(self.conv1(B))  # torch.Size([bs, 384, 768])
        B = self.conv2(B)  # torch.Size([bs, 1, 768])
        B = B.squeeze(1)  # torch.Size([bs, 768])
        B = self.batchnorm2(B)  # torch.Size([bs, 768])
        return B

    # module 2: predict from original amino acid sequence (aaindex1 feature)
    def aa_predict(self, lchain, hchain, antigen):
        # 使用矩阵分解
        l_ag = self.fusion1(lchain, antigen)
        h_ag = self.fusion2(hchain, antigen)
        x = torch.cat([l_ag, h_ag], 1)  # torch.Size([bs, 256*2])
        x = self.batchnorm(x)  # torch.Size([bs, 512])
        x = F.dropout(x, p=0.5, training=self.training)
        return x

    def predict(self, lchain_aaindex, hchain_aaindex, ag_aaindex, lchain_embedding, hchain_embedding, ag_embedding):
        phat_1 = self.map_predict(lchain_embedding, hchain_embedding, ag_embedding)
        phat_2 = self.aa_predict(lchain_aaindex, hchain_aaindex, ag_aaindex)
        x = torch.cat([phat_1, phat_2], 1)  # torch.Size([bs, 768+512])
        x = (self.fc2(x))
        x = (self.fc3(x))
        x = (self.fc4(x))
        phat = self.activation(x)
        return phat

    def forward(self, lchain_id, hchain_id, ag_id, aaindex_feature, lchain_embedding, hchain_embedding, ag_embedding):
        return self.predict(lchain_id, hchain_id, ag_id, aaindex_feature, lchain_embedding, hchain_embedding, ag_embedding)


class FusionModel(nn.Module):
    def __init__(self, input_dim, fusion_dim):  # [20*256, 256]
        super(FusionModel, self).__init__()
        self.fusion_matrix = nn.Linear(input_dim, fusion_dim)
    def forward(self, x1, x2):
        x1_flat = x1.view(x1.size(0), -1).float()
        x2_flat = x2.view(x2.size(0), -1).float()
        x1_transformed = self.fusion_matrix(x1_flat)
        x2_transformed = self.fusion_matrix(x2_flat)

        fused_tensor = x1_transformed + x2_transformed
        return fused_tensor

