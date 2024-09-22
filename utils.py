import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import torch
import logging
import os.path as osp


def seed_setting(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def zero2eps(x):

    x[x == 0] = 1
    return x

def compute_centers(x, psedo_labels, num_cluster):
    n_samples = x.size(0)
    if len(psedo_labels.size()) > 1:
        # 如果伪标签 psedo_labels 是一个二维张量，则将其转置得到一个形状为 (n_samples, num_cluster) 的张量
        weight = psedo_labels.T
    else:
        # 如果伪标签 psedo_labels 是一个一维张量，则创建一个形状为 (num_cluster, n_samples) 的零张量 weight，
        # 并将其中第 psedo_labels[i] 行、第 i 列的元素设置为 1，表示第 i 个样本属于第 psedo_labels[i] 个聚类
        weight = torch.zeros(num_cluster, n_samples).to(x)  # L, N
        weight[psedo_labels, torch.arange(n_samples)] = 1

    weight = weight.float()
    # 对 weight 进行 L1 归一化，即将每一行的元素值都除以该行的元素之和，以确保每个聚类的权重之和为 1
    weight = F.normalize(weight, p=1, dim=1)  # l1 normalization
    # 通过矩阵乘法 torch.mm(weight, x) 将 weight 和 x 相乘，得到每个聚类的样本特征的加权平均值，即聚类中心
    centers = torch.mm(weight, x)
    # 对聚类中心进行 L2 归一化，以确保每个聚类中心向量的长度为1
    centers = F.normalize(centers, dim=1)

    return centers




def compute_cluster_loss(q_centers, k_centers, temperature, psedo_labels, num_cluster):
    # 首先计算当前轮次聚类中心之间的相似度矩阵 d_q
    d_q = q_centers.mm(q_centers.T) / temperature
    # 计算当前轮次聚类中心和历史轮次聚类中心之间的相似度向量 d_k
    d_k = (q_centers * k_centers).sum(dim=1) / temperature
    d_q = d_q.float()
    # 将 d_k 的值分别赋给 d_q 的对角线上的元素，
    # 以确保每个聚类中心与历史轮次中与之对应的聚类中心之间的相似度得到正确计算
    d_q[torch.arange(num_cluster), torch.arange(num_cluster)] = d_k

    # 找出伪标签 psedo_labels 中没有被分配的聚类中心的下标，存储在 zero_classes 中
    zero_classes = torch.nonzero(torch.sum(psedo_labels, dim=0) == 0).squeeze()

    # 将没有分配到数据点的聚类中心之间的相似度设置为一个大负数，以便在 softmax 操作中将它们的概率值设为接近于 0 的极小数
    mask = torch.zeros((num_cluster, num_cluster), dtype=torch.bool, device=d_q.device)
    mask[:, zero_classes] = 1
    d_q.masked_fill_(mask, -10)

    # 获取 d_q 矩阵的对角线上的元素，存储在变量 pos 中
    pos = d_q.diag(0)
    mask = torch.ones((num_cluster, num_cluster))
    # 将 mask 张量对角线上的元素全部设置为 0，即生成一个对角线上的元素为 0，其余元素为 1 的矩阵，再转成布尔型
    # 这样生成的 mask 矩阵将在后续的计算中用于掩盖掉 d_q 矩阵中对角线上的元素 pos，从而避免对已经确定的聚类中心进行重新分配
    mask = mask.fill_diagonal_(0).bool()

    # 用于计算聚类中心之间的 softmax 交叉熵损失
    neg = d_q[mask].reshape(-1, num_cluster - 1)
    loss = - pos + torch.logsumexp(torch.cat([pos.reshape(num_cluster, 1), neg], dim=1), dim=1)
    loss[zero_classes] = 0.
    # loss = loss.sum() / (num_cluster - len(zero_classes))
    if zero_classes.numel() < num_cluster:
        loss = loss.sum() / (num_cluster - zero_classes.numel())
    else:
        loss = 0.

    return loss


def normalize(affinity):
    col_sum = zero2eps(np.sum(affinity, axis=1)[:, np.newaxis])
    row_sum = zero2eps(np.sum(affinity, axis=0))
    out_affnty = affinity/col_sum
    in_affnty = np.transpose(affinity/row_sum)
    return in_affnty, out_affnty


def affinity_tag_multi(tag1: np.ndarray, tag2: np.ndarray):
    aff = np.matmul(tag1, tag2.T)
    affinity_matrix = np.float32(aff)
    affinity_matrix = 1 / (1 + np.exp(-affinity_matrix))
    affinity_matrix = 2 * affinity_matrix - 1
    in_aff, out_aff = normalize(affinity_matrix)
    return in_aff, out_aff, affinity_matrix


def calculate_map(qu_B, re_B, qu_L, re_L):
    num_query = qu_L.shape[0]
    map = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        count = np.linspace(1, tsum, tsum)
        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        map = map + map_
    map = map / num_query
    return map


def calculate_top_map(qu_B, re_B, qu_L, re_L, topk):
    num_query = qu_L.shape[0]
    topkmap = 0
    for iter in range(num_query):
        gnd = (np.dot(qu_L[iter, :], re_L.transpose()) > 0).astype(np.float32)
        hamm = calculate_hamming(qu_B[iter, :], re_B)
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, int(tsum))
        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap



def calculate_hamming(B1, B2):
    leng = B2.shape[1] 
    distH = 0.5 * (leng - np.dot(B1, B2.transpose()))
    return distH


def qmi_loss(code, targets, eps=1e-8):

    code = code / (torch.sqrt(torch.sum(code ** 2, dim=1, keepdim=True)) + eps)
    Y = torch.mm(code, code.t())
    Y = 0.5 * (Y + 1)
    targets = targets.float()
    D = targets.mm(targets.transpose(0, 1)) > 0
    D = D.type(torch.cuda.FloatTensor)

    M = D.size(1) ** 2 / torch.sum(D)

    Qy_in = (D * Y - 1) ** 2
    Qy_btw = (1.0 / M) * Y ** 2
    loss = torch.sum(Qy_in + Qy_btw)
    return loss



def calc_hamming_dist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.t()))
    return distH

def p_topK(qB, rB, query_label, retrieval_label, K=None):
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    num_query = query_label.shape[0]
    p = [0] * len(K)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm)[1][:total]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p

def p_topK2(qB, rB, query_label, retrieval_label, K):
    num_query = query_label.shape[0]
    p = [0] * len(K)
    query_label = torch.Tensor(query_label)
    retrieval_label = torch.Tensor(retrieval_label)
    qB = torch.Tensor(qB)
    rB = torch.Tensor(rB)
    for iter in range(num_query):
        gnd = (query_label[iter].unsqueeze(0).mm(retrieval_label.t()) > 0).float().squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hamming_dist(qB[iter, :], rB).squeeze()
        hamm = torch.Tensor(hamm)
        for i in range(len(K)):
            total = min(K[i], retrieval_label.shape[0])
            ind = torch.sort(hamm).indices[:int(total)]
            gnd_ = gnd[ind]
            p[i] += gnd_.sum() / total
    p = torch.Tensor(p) / num_query
    return p



def logger(fileName='log'):
    logger = logging.getLogger('train')
    logger.setLevel(logging.INFO)
    log_name = str(fileName) + '.log'
    log_dir = './logs'
    txt_log = logging.FileHandler(osp.join(log_dir, log_name))
    txt_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    txt_log.setFormatter(formatter)
    logger.addHandler(txt_log)
    stream_log = logging.StreamHandler()
    stream_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    stream_log.setFormatter(formatter)
    logger.addHandler(stream_log)
    return logger