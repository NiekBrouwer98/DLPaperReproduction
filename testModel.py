import numpy as np
import os
import torch
import sys
from model import ABE_M
from sampler_excelfile import MetricData, SourceSampler
from torch.utils.data import DataLoader


def test(test_loader, model, device):
    with torch.no_grad():
        model.eval()
        embedding_vec = []
        for batch_idx, batch in enumerate(test_loader):
            data, target = batch
            data = data.to(device)
            outputs = model(data)
            if batch_idx == 0:
                for i in range(len(outputs)):
                    output = outputs[i].cpu().numpy()
                    embedding_vec.append(output)
                labels = np.array(target)
            else:
                for i in range(len(outputs)):
                    output = outputs[i].cpu().numpy()
                    embedding_vec[i] = np.concatenate((embedding_vec[i], output))
                labels = np.concatenate((labels, np.array(target)))

    return embedding_vec, labels


def Accuracy(embedding_vec, labels):
    print(labels)

    def _get_sim_matrix(embedding_vec, num_vec):
        sim_matrix = np.zeros((num_vec, num_vec))
        for i in range(len(embedding_vec)):
            sim_matrix += np.matmul(embedding_vec[i], embedding_vec[i].T)
        if len(embedding_vec) > 1:
            sim_matrix = sim_matrix / len(embedding_vec)
        return sim_matrix

    k_list = [1,2,4,8]
    # k_list = [1]
    max_k = max(k_list)
    num_vec = embedding_vec[0].shape[0]
    print('total num of test image: {}'.format(num_vec))
    sim_mat = _get_sim_matrix(embedding_vec, num_vec)
    result = np.zeros((num_vec, max_k + 1))

    for i in range(num_vec):
        temp_index = np.argsort(-1 * sim_mat[i])[:max_k + 1]
        for j in range(max_k + 1):
            result[i][j] = labels[temp_index[j]]

    predicts = result[:, 1]
    n_p = sum(labels == predicts) / num_vec
    # num_class = len(set(labels))
    num_class = 200
    cnf_mat = np.zeros((num_class, num_class))
    for i in range(len(labels)):
        cnf_mat[int(labels[i])-1][int(predicts[i])-1] += 1

    e = cnf_mat.diagonal()
    n_w = (e / (cnf_mat.sum(axis=1) + 0.00000001)).mean()
    n_total = n_w * n_p
    accuracy = [(n_p, 'np'), (n_w, 'nw'), (n_total, 'n_total')]

    recall = []
    for k in k_list:
        total = 0
        for i in range(num_vec):
            temp_list = result[i][1:k + 1].tolist()
            if labels[i] in temp_list:
                total += 1
        recall.append((total / num_vec, k))
        print('Recall@k:{:.4f}, k={}'.format(total / num_vec, k))

    return accuracy


def loadTestData():
    datatest = MetricData(data_root='CUB_100_test',
                          anno_file=r'./data/annos_testdataset.xlsx',
                          idx_file='./data/idx_testset.pkl',
                          return_fn=True)

    samplertest = SourceSampler(datatest)
    testdata_loader = torch.utils.data.DataLoader(datatest, batch_sampler=samplertest)

    return testdata_loader


def main(modelpath):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 32
    data = loadTestData()
    model = ABE_M()
    model.load_state_dict(torch.load(modelpath))
    model.to(device)

    embedding_vec, labels = test(data, model, device)
    accuracy_his = Accuracy(embedding_vec, labels)
    print(accuracy_his)


if __name__ == '__main__':
    main('./model/score_0.2646.pth')
