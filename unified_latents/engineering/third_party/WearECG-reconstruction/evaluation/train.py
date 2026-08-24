"""
PTB-XL多标签多分类任务示例代码
本代码基于上传的ResNet1D实现（Hong, 2019），
目标是对PTB-XL数据集中6种心电诊断（Normal, Afib, CLBBB, CRBBB, LVH, RVH）进行多标签分类。
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, roc_auc_score
import random

# 导入上传的ResNet1D实现
from resnet1d import ResNet1D

# 固定随机种子，保证实验结果可复现
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# 定义目标类别及映射（目标类别顺序对应多标签向量的各维）
TARGET_CLASSES = ['NORM', 'AFIB', 'CLBBB', 'CRBBB', 'LVH', 'RVH']
# TARGET_CLASSES = [
#     '1AVB', '2AVB', '3AVB', 'ABQRS', 'AFIB', 'AFLT', 'ALMI', 'AMI', 'ANEUR', 'ASMI',
#     'BIGU', 'CLBBB', 'CRBBB', 'DIG', 'EL', 'HVOLT', 'ILBBB', 'ILMI', 'IMI', 'INJAL',
#     'INJAS', 'INJIL', 'INJIN', 'INJLA', 'INVT', 'IPLMI', 'IPMI', 'IRBBB', 'ISCAL', 'ISCAN',
#     'ISCAS', 'ISCIL', 'ISCIN', 'ISCLA', 'ISC_', 'IVCD', 'LAFB', 'LAO/LAE', 'LMI', 'LNGQT',
#     'LOWT', 'LPFB', 'LPR', 'LVH', 'LVOLT', 'NDT', 'NORM', 'NST_', 'NT_', 'PAC', 'PACE',
#     'PMI', 'PRC(S)', 'PSVT', 'PVC', 'QWAVE', 'RAO/RAE', 'RVH', 'SARRH', 'SBRAD', 'SEHYP',
#     'SR', 'STACH', 'STD_', 'STE_', 'SVARR', 'SVTAC', 'TAB_', 'TRIGU', 'VCLVH', 'WPW'
# ]

n_classes = len(TARGET_CLASSES)

def load_and_filter_data():
    X_train = np.load("/data2/2shared/fangxiaocheng/ecg_ptbxl_benchmarking/data/PTBXL/all/data/train_data.npy")
    print(X_train.shape)
    y_train_encoded = np.load("/data2/2shared/fangxiaocheng/ecg_ptbxl_benchmarking/data/PTBXL/all/data/train_labels.npy")
    X_test = np.load("/data2/2shared/fangxiaocheng/ecg_ptbxl_benchmarking/data/PTBXL/all/data/test_data.npy")
    print(X_test.shape)
    y_test_encoded = np.load("/data2/2shared/fangxiaocheng/ecg_ptbxl_benchmarking/data/PTBXL/all/data/test_labels.npy")
    
    # 加载mlb.pkl解码标签
    mlb_pkl_path = "/data2/2shared/fangxiaocheng/ecg_ptbxl_benchmarking/data/PTBXL/all/data/mlb.pkl"
    with open(mlb_pkl_path, "rb") as f:
        mlb = pickle.load(f)
        
    # 使用mlb解码得到标签列表，返回的是一个tuple列表，每个tuple包含该样本的所有标签
    y_train_decoded = mlb.inverse_transform(y_train_encoded)
    y_test_decoded = mlb.inverse_transform(y_test_encoded)
    
    def process_labels(X, y):
        filtered_indices = []
        multi_hot_labels = []
        for i, labels in enumerate(y):
            # labels可能为tuple类型，转换为list便于判断
            labels = list(labels)
            # print(labels)
            # 生成目标类别对应的多热向量
            present = [1 if t in labels else 0 for t in TARGET_CLASSES]
            # print(present)
            # 仅保留至少包含一个目标标签的样本
            if sum(present) > 0:
                filtered_indices.append(i)
                multi_hot_labels.append(present)
        X_filtered = X[filtered_indices]
        y_filtered = np.array(multi_hot_labels, dtype=np.float32)
        return X_filtered, y_filtered

    X_train_filtered, y_train_filtered = process_labels(X_train, y_train_decoded)
    X_test_filtered, y_test_filtered = process_labels(X_test, y_test_decoded)
    
    return X_train_filtered, y_train_filtered, X_test_filtered, y_test_filtered


def load_and_filter_generate_data():
    # X_test = np.load("/data2/2shared/fangxiaocheng/ecg_reconstruction/samples/ekgan/overall_fake_data_ptbxl.npy")
    # X_test = np.load("/data1/1shared/jinjiarui/run/MLA-Diffusion/results/cond_syn/PTBXL/samples/overall_fake_data.npy")
    # X_test = np.load("/data2/2shared/fangxiaocheng/MCMA/samples/PTBXL/overall_fake_data.npy")
    X_test = np.load("/data2/2shared/fangxiaocheng/ecg_reconstruction/samples/pix2pix/overall_fake_data_ptbxl.npy")
    # X_test = np.load("/data2/2shared/fangxiaocheng/ecg_reconstruction/samples/cyclegan/overall_fake_data_ptbxl.npy")
    y_test_encoded = np.load("/data2/2shared/fangxiaocheng/ecg_ptbxl_benchmarking/data/PTBXL/all/data/test_labels.npy")
    
    # 加载mlb.pkl解码标签
    mlb_pkl_path = "/data2/2shared/fangxiaocheng/ecg_ptbxl_benchmarking/data/PTBXL/all/data/mlb.pkl"
    with open(mlb_pkl_path, "rb") as f:
        mlb = pickle.load(f)
        
    # 使用mlb解码得到标签列表，返回的是一个tuple列表，每个tuple包含该样本的所有标签
    y_test_decoded = mlb.inverse_transform(y_test_encoded)
    
    def process_labels(X, y):
        filtered_indices = []
        multi_hot_labels = []
        for i, labels in enumerate(y):
            # labels可能为tuple类型，转换为list便于判断
            labels = list(labels)
            # print(labels)
            # 生成目标类别对应的多热向量
            present = [1 if t in labels else 0 for t in TARGET_CLASSES]
            # print(present)
            # 仅保留至少包含一个目标标签的样本
            if sum(present) > 0:
                filtered_indices.append(i)
                multi_hot_labels.append(present)
        X_filtered = X[filtered_indices]
        y_filtered = np.array(multi_hot_labels, dtype=np.float32)
        return X_filtered, y_filtered

    X_test_filtered, y_test_filtered = process_labels(X_test, y_test_decoded)
    
    return X_test_filtered, y_test_filtered

# 定义适用于多标签任务的数据集类
class MyMultiLabelDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        # 标签采用浮点型张量
        return torch.tensor(self.data[index], dtype=torch.float), torch.tensor(self.label[index], dtype=torch.float)

    def __len__(self):
        return len(self.data)

def train(model, device, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx+1) % 10 == 0:
            print(f"Epoch {epoch} Batch {batch_idx+1}/{len(train_loader)} Loss: {loss.item():.4f}")
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch} Average Training Loss: {avg_loss:.4f}")
    return avg_loss

def evaluate(model, device, data_loader, threshold=0.5):
    """
    多标签评价函数：模型输出经过sigmoid激活后，与阈值比较以得到二值化预测。
    同时返回预测的概率，用于后续AUC-ROC计算。
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    with torch.no_grad():
        for data, target in data_loader:
            data = data.to(device)
            output = model(data)
            prob = torch.sigmoid(output).cpu().numpy()
            preds = (prob > threshold).astype(int)
            all_preds.extend(preds)
            all_probs.extend(prob)
            all_targets.extend(target.numpy().astype(int))
    return all_targets, all_preds, all_probs

def main():
    # 加载并筛选数据，同时将标签通过mlb.pkl解码后转换为多热编码
    X_train, y_train, X_test, y_test = load_and_filter_data()
    print(f"训练集样本数：{len(y_train)}，测试集样本数：{len(y_test)}")
    
    # 构建多标签数据集与数据加载器
    train_dataset = MyMultiLabelDataset(X_train, y_train)
    test_dataset = MyMultiLabelDataset(X_test, y_test)
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("使用设备：", device)
    
    # 定义模型参数（可根据实际情况进行调整）
    in_channels = X_train.shape[1]  # 假设为12导联数据
    base_filters = 64
    kernel_size = 7
    stride = 1
    groups = 1
    n_block = 16
    
    model = ResNet1D(in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, verbose=False)
    model = model.to(device)
    
    # 使用适合多标签任务的损失函数
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 训练过程
    epochs = 30
    weight_file = "/data2/2shared/fangxiaocheng/ecg_ptbxl_benchmarking/resnet1d/checkpoints/resnet1d_ptbxl_model.pth"
    
    # 检查是否存在预先保存的模型权重
    if os.path.exists(weight_file):
        print(f"检测到模型权重文件 {weight_file}，直接加载权重。")
        model.load_state_dict(torch.load(weight_file, map_location=device))
    else:
        for epoch in range(1, epochs+1):
            train(model, device, train_loader, criterion, optimizer, epoch)
        torch.save(model.state_dict(), weight_file)
        print(f"模型权重已保存至：{weight_file}")

    # 在原始测试集上评估模型
    print("在原始测试集上评估模型：")
    true_labels, pred_labels, probs = evaluate(model, device, test_loader, threshold=0.5)
    print(classification_report(true_labels, pred_labels, target_names=TARGET_CLASSES))
    auc_score = roc_auc_score(np.array(true_labels), np.array(probs), average='macro')
    print(f"AUC-ROC: {auc_score:.4f}")

    # 生成的十二导联测试数据，并构建对应DataLoader
    X_test_generated, y_test_label= load_and_filter_generate_data()
    generated_test_dataset = MyMultiLabelDataset(X_test_generated, y_test_label)
    generated_test_loader = DataLoader(generated_test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("生成的十二导联测试数据：")
    true_labels_gen, pred_labels_gen, probs_gen = evaluate(model, device, generated_test_loader, threshold=0.5)
    print(classification_report(true_labels_gen, pred_labels_gen, target_names=TARGET_CLASSES))
    auc_score_gen = roc_auc_score(np.array(true_labels_gen), np.array(probs_gen), average='macro')
    print(f"AUC-ROC (生成数据): {auc_score_gen:.4f}")

if __name__ == "__main__":
    main()
