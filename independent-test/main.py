import pickle
import torch
import numpy as np
import random
import os
import argparse
from model import *
import timeit
os.chdir(os.path.dirname(os.path.abspath(__file__)))
def load_tensor(file_name, dtype):
    return [dtype(d).to(device) for d in np.load(file_name + '.npy')]

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def init_seed(SEED = 2021):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True

from sklearn.model_selection import StratifiedKFold
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='independent-test')
    parser.add_argument('--model_name', type=str, default='independent-test', help='The name of models')
    parser.add_argument('--protein_dim', type=int, default=100, help='embedding dimension of proteins')
    parser.add_argument('--atom_dim', type=int, default=34, help='embedding dimension of atoms')
    parser.add_argument('--hid_dim', type=int, default=64, help='embedding dimension of hidden layers')
    parser.add_argument('--n_layers', type=int, default=3, help='layer count of networks')
    parser.add_argument('--n_heads', type=int, default=8, help='the head count of self-attention')
    parser.add_argument('--pf_dim', type=int, default=256, help='dimension of feedforward neural network')
    parser.add_argument('--dropout', type=float, default=0.2, help='the ratio of Dropout')
    parser.add_argument('--batch', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--iteration', type=int, default=100, help='the iteration for training')
    parser.add_argument('--n_folds', type=int, default=5, help='the fold count for cross-entropy')
    parser.add_argument('--seed', type=int, default=2021, help='the random seed')
    parser.add_argument('--kernel_size', type=int, default=9, help='the kernel size of Conv1D in transformer')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open('./data/train.pickle',"rb") as f:
        data = pickle.load(f)
    data_train = shuffle_dataset(data, 1234)

    with open('./data/test.pickle',"rb") as f:
        data = pickle.load(f)
    data_test = shuffle_dataset(data, 1234)
    labels = [i[-3] for i in data_train]
    skf = StratifiedKFold(n_splits=args.n_folds)

    init_seed(args.seed)
    results = np.array([0.0]*4)
    test_auc, test_prc, test_pre, test_recall = 0.0, 0.0, 0.0, 0.0
    for fold, (train_idx, val_idx) in enumerate(skf.split(data_train, labels)):
        dataset_train = [data_train[idx] for idx in train_idx]
        dataset_dev = [data_train[idx] for idx in val_idx]

        model = Predictor(args.hid_dim, args.n_layers, args.kernel_size, args.n_heads, args.pf_dim, args.dropout, device, args.atom_dim, args.protein_dim)
        model.to(device)
        trainer = Trainer(model, args.lr, args.weight_decay, args.batch, len(dataset_train))
        tester = Tester(model)

        file_AUCs = f'./result/{args.model_name}_{fold}.txt'
        file_auc_test = f'./result/test_{args.model_name}_{fold}.txt'
        file_model = f'./model/{args.model_name}_{fold}.pt'

        AUCs = ('Epoch\tTime(sec)\tLoss_train\tAUC_dev\tPRC_dev\tPrecison_dev\tRecall_dev')
        with open(file_AUCs, 'w') as f:
            f.write(AUCs + '\n')

        """Start training."""
        print('Training...')
        print(AUCs)
        start = timeit.default_timer()
        max_AUC_dev = 0
        for epoch in range(1, args.iteration+1):
            loss_train = trainer.train(dataset_train, device)
            AUC_dev, PRC_dev, PRE_dev, REC_dev = tester.test(dataset_dev)
            end = timeit.default_timer()
            time = end - start

            AUCs = [epoch, time//60, loss_train, AUC_dev,PRC_dev, PRE_dev, REC_dev]
            tester.save_AUCs(AUCs, file_AUCs)
            if AUC_dev > max_AUC_dev:
                tester.save_model(model, file_model)
                max_AUC_dev = AUC_dev

                test_auc, test_prc, test_pre, test_recall = tester.test(data_test)
                tester.save_AUCs([epoch, test_auc, test_prc, test_pre, test_recall], file_auc_test)
                print(f'Test ---> AUC: {test_auc}, PRC: {test_prc}')
            print('\t'.join(map(str, AUCs)))

            results += np.array([test_auc, test_prc, test_pre, test_recall])
        results /= args.n_folds
    print('\t'.join(map(str, results)) + '\n')