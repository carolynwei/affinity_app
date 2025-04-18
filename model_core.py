# model_core.py
import torch
import numpy as np
from tape import TAPETokenizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

# ============ 加载模型 ============
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrain_model_path = './pretrain_bert.models'
model_path = './model1231_epoch30.pth'
aaindex_path = './aaindex_pca.csv'

pretrain_model = torch.load(pretrain_model_path, weights_only=False)
pretrain_model.to(device)
pretrain_model.eval()
for param in pretrain_model.parameters():
    param.requires_grad = False

main_model = torch.load(model_path, weights_only=False)
main_model.to(device)
main_model.eval()

# ============ 特征构建函数 ============
def get_aaindex_feature(seqs, aaindex_path, device, max_len=256):
    df = pd.read_csv(aaindex_path, header=0)
    scaler = MinMaxScaler()
    df[df.columns[1:]] = scaler.fit_transform(df[df.columns[1:]])
    groups = df.groupby('AA')
    results_dict = {
        name: [row[df.columns[1:21]].tolist() for _, row in group.iterrows()]
        for name, group in groups
    }

    aaindex_feature = []
    for seq in seqs:
        feat = [torch.tensor(results_dict[aa][0]) if aa in results_dict else torch.zeros(20) for aa in seq]
        feat = (feat + [torch.zeros(20)] * (max_len - len(feat)))[:max_len]
        aaindex_feature.append(torch.stack(feat).unsqueeze(0).to(device))
    return aaindex_feature

def get_bert_feature(seqs, device):
    tokenizer = TAPETokenizer(vocab='iupac')
    seq_embeddings = []
    for seq in seqs:
        token_ids = torch.tensor([tokenizer.encode(seq)], dtype=torch.long).to(device)
        seq_embeddings.append(pretrain_model(token_ids)[1][0].unsqueeze(0))
    return seq_embeddings

# ============ 核心函数 ============
def predict_affinity(seq_light: str, seq_heavy: str, seq_antigen: str) -> float:
    seqs = [seq_light, seq_heavy, seq_antigen]
    aaindex_feature = get_aaindex_feature(seqs, aaindex_path, device)
    bert_feature = get_bert_feature(seqs, device)

    with torch.no_grad():
        pred = main_model.predict(
            aaindex_feature[0], aaindex_feature[1], aaindex_feature[2],
            bert_feature[0], bert_feature[1], bert_feature[2]
        )
        p_hat = (pred * (16.9138 - 5.0400)) + 5.0400  # 反归一化
    return p_hat.item()
