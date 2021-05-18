import torch
import numpy as np
import random
import os
from model import *
import timeit
import argparse
import pickle
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def init_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Kinase Traing')
    parser.add_argument('--model_name', type=str, default='kinase', help='The name of models')
    parser.add_argument('--protein_dim', type=int, default=100, help='embedding dimension of proteins')
    parser.add_argument('--atom_dim', type=int, default=34, help='embedding dimension of atoms')
    parser.add_argument('--hid_dim', type=int, default=64, help='embedding dimension of hidden layers')
    parser.add_argument('--n_layers', type=int, default=3, help='layer count of networks')
    parser.add_argument('--n_heads', type=int, default=8, help='the head count of self-attention')
    parser.add_argument('--pf_dim', type=int, default=256, help='dimension of feedforward neural network')
    parser.add_argument('--dropout', type=float, default=0.2, help='the ratio of Dropout')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight decay')
    parser.add_argument('--iteration', type=int, default=50, help='the iteration for training')
    parser.add_argument('--n_folds', type=int, default=5, help='the fold count for cross-entropy')
    parser.add_argument('--seed', type=int, default=2021, help='the random seed')
    parser.add_argument('--kernel_size', type=int, default=9, help='the kernel size of Conv1D in transformer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    init_seed(args.seed)

    with open('./data/train.pickle',"rb") as f:
        data = pickle.load(f)
    dataset = shuffle_dataset(data, args.seed)
    dataset_train, dataset_dev = split_dataset(dataset, 0.8)
    N_SAMPELES = len(dataset_train)

    with open('./data/test.pickle',"rb") as f:
        data = pickle.load(f)
    dataset_test = shuffle_dataset(data, args.seed)

    model = Predictor(args.hid_dim, args.n_layers, args.kernel_size, args.n_heads, args.pf_dim, args.dropout, device, args.atom_dim, args.protein_dim)
    model.to(device)
    trainer = Trainer(model, args.lr, args.weight_decay, args.batch, N_SAMPELES)
    tester = Tester(model)

    file_AUCs = f'./result/{args.model_name}.txt'
    file_auc_test = f'./result/test_{args.model_name}.txt'
    file_model = f'./model/{args.model_name}.pt'
    AUC = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tPRC_dev\tPrecision_dev\tRecall_dev')
    with open(file_AUCs, 'w') as f:
        f.write(AUC + '\n')

    print('Training...')
    print(AUC)
    start = timeit.default_timer()
    max_AUC_dev = 0
    best_epoch, best_AUC_test, best_AUPR_test, best_precision_test, best_recall_test = 0.0, 0.0, 0.0, 0.0, 0.0
    for epoch in range(1, args.iteration + 1):
        loss_train = trainer.train(dataset_train, device)
        AUC_dev, PRC_dev, pre_dev, recall_dev = tester.test(dataset_dev)
        end = timeit.default_timer()
        time = end - start

        AUCs = [epoch, time//60, loss_train, AUC_dev, PRC_dev, pre_dev, recall_dev]
        tester.save_AUCs(AUCs, file_AUCs)
        if AUC_dev > max_AUC_dev:
            tester.save_model(model, file_model)
            max_AUC_dev = AUC_dev
            test_auc, test_prc, test_pre, test_recall = tester.test(dataset_test)
            tester.save_AUCs([epoch, test_auc, test_prc, test_pre, test_recall], file_auc_test)
            best_epoch = epoch
            best_AUC_test = test_auc
            best_AUPR_test = test_prc
            best_precision_test = test_pre
            best_recall_test = test_recall
            print(f'Test ---> AUC: {AUC}, PRC: {test_prc}')
        print('\t'.join(map(str, AUCs)))
    print('\t'.join(map(str, [tra_idx, best_epoch, best_auc_test, best_aupr_test, best_precision_test, best_recall_test])) + '\n')
