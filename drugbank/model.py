import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
import sys
sys.path.append('..')
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score,precision_recall_curve, auc
from utils.lookahead import Lookahead
class SelfAttention(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        assert hid_dim % n_heads == 0
        self.w_q = nn.Linear(hid_dim, hid_dim)
        self.w_k = nn.Linear(hid_dim, hid_dim)
        self.w_v = nn.Linear(hid_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, hid_dim)
        self.do = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads])).to(device)

    def forward(self, query, key, value, mask=None):
        bsz = query.shape[0]
        Q = self.w_q(query)
        K = self.w_k(key)
        V = self.w_v(value)

        Q = Q.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        K = K.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)
        V = V.view(bsz, -1, self.n_heads, self.hid_dim // self.n_heads).permute(0, 2, 1, 3)

        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = self.do(F.softmax(energy, dim=-1))
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
        x = self.fc(x)
        return x

class Encoder(nn.Module):
    def __init__(self, protein_dim, hid_dim, n_layers, kernel_size , dropout, device):
        super().__init__()
        assert kernel_size % 2 == 1
        self.input_dim = protein_dim
        self.hid_dim = hid_dim
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.n_layers = n_layers
        self.device = device
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.convs = nn.ModuleList([nn.Conv1d(hid_dim, 2*hid_dim, kernel_size, padding=(kernel_size-1)//2) for _ in range(self.n_layers)])   # convolutional layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.input_dim, self.hid_dim)
        self.gn = nn.GroupNorm(8, hid_dim * 2)
        self.ln = nn.LayerNorm(hid_dim)

    def forward(self, protein):
        conv_input = self.fc(protein)
        conv_input = conv_input.permute(0, 2, 1)
        for i, conv in enumerate(self.convs):
            conved = conv(self.dropout(conv_input))
            conved = F.glu(conved, dim=1)
            conved = (conved + conv_input) * self.scale
            conv_input = conved

        conved = conved.permute(0, 2, 1)
        conved = self.ln(conved)
        return conved

class PositionwiseFeedforward(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.hid_dim = hid_dim
        self.pf_dim = pf_dim
        self.fc_1 = nn.Conv1d(hid_dim, pf_dim, 1)  # convolution neural units
        self.fc_2 = nn.Conv1d(pf_dim, hid_dim, 1)  # convolution neural units
        self.do = nn.Dropout(dropout)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.do(F.relu(self.fc_1(x)))
        x = self.fc_2(x)
        x = x.permute(0, 2, 1)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device):
        super().__init__()

        self.ln = nn.LayerNorm(hid_dim)
        self.sa = self_attention(hid_dim, n_heads, dropout, device)
        self.ea = self_attention(hid_dim, n_heads, dropout, device)
        self.pf = positionwise_feedforward(hid_dim, pf_dim, dropout)
        self.do = nn.Dropout(dropout)

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = self.ln(trg + self.do(self.sa(trg, trg, trg, trg_mask)))
        trg = self.ln(trg + self.do(self.ea(trg, src, src, src_mask)))
        trg = self.ln(trg + self.do(self.pf(trg)))
        return trg

class Decoder(nn.Module):
    def __init__(self, embed_dim, hid_dim, n_layers, n_heads, pf_dim, decoder_layer, self_attention,
                 positionwise_feedforward, dropout, device):
        super(Decoder, self).__init__()
        self.ft = nn.Linear(embed_dim, hid_dim)
        self.layers = nn.ModuleList(
            [decoder_layer(hid_dim, n_heads, pf_dim, self_attention, positionwise_feedforward, dropout, device)
             for _ in range(n_layers)])
        self.dropout = dropout
        self.device = device

    def forward(self, trg, src, trg_mask=None, src_mask=None):
        trg = F.dropout(self.ft(trg), p=self.dropout)
        for layer in self.layers:
            trg = layer(trg, src, trg_mask, src_mask)
        return trg #[bs, seq_len, hid_dim]

class TextCNN(nn.Module):
    def __init__(self, embed_dim, hid_dim, kernels=[3, 5, 7], dropout_rate=0.5):
        super(TextCNN, self).__init__()
        padding1 = (kernels[0] - 1) // 2
        padding2 = (kernels[1] - 1) // 2
        padding3 = (kernels[2] - 1) // 2

        self.conv1 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[0], padding=padding1),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[1], padding=padding2),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(embed_dim, hid_dim, kernel_size=kernels[2], padding=padding3),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
            nn.Conv1d(hid_dim, hid_dim, kernel_size=kernels[2], padding=padding3),
            nn.BatchNorm1d(hid_dim),
            nn.PReLU(),
        )

        self.conv = nn.Sequential(
            nn.Linear(hid_dim*len(kernels), hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
            nn.BatchNorm1d(hid_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hid_dim, hid_dim),
        )
    def forward(self, protein):
        protein = protein.permute([0, 2, 1])  #[bs, hid_dim, seq_len]
        features1 = self.conv1(protein)
        features2 = self.conv2(protein)
        features3 = self.conv3(protein)
        features = torch.cat((features1, features2, features3), 1)  #[bs, hid_dim*3, seq_len]
        features = features.max(dim=-1)[0]  #[bs, hid_dim*3]
        return self.conv(features)

class Predictor(nn.Module):
    def __init__(self, hid_dim, n_layers, kernel_size, n_heads, pf_dim, dropout, device, atom_dim=34, protein_dim=100):
        super(Predictor, self).__init__()
        id2smi, smi2id, smi_embed = np.load('../data/pretrain_embed/smi2vec.npy')
        id2prot, prot2id, prot_embed = np.load('../data/pretrain_embed/prot2vec.npy')

        self.dropout = dropout
        self.device = device
        self.prot_embed = nn.Embedding(len(prot_embed)+1, len(prot_embed[0]), padding_idx=0)
        self.prot_embed.data = prot_embed
        for param in self.prot_embed.parameters():
            param.requires_grad = False

        self.smi_embed = nn.Embedding(len(smi_embed)+1, len(smi_embed[0]), padding_idx=0)
        self.smi_embed.data = smi_embed
        for param in self.smi_embed.parameters():
            param.requires_grad = False
        print(f'prot Embed: {len(prot_embed)},  smi Embed: {len(smi_embed)}')

        # protein encoding, target decoding
        self.enc_prot = Encoder(len(prot_embed[0]), hid_dim, n_layers, kernel_size, dropout, device)
        self.dec_smi = Decoder(len(smi_embed[0]), hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

        # target  encoding, protein decoding
        self.enc_smi = Encoder(len(smi_embed[0]), hid_dim, n_layers, kernel_size, dropout, device)
        self.dec_prot = Decoder(len(prot_embed[0]), hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

        # TextCNN
        self.prot_textcnn = TextCNN(100, hid_dim)

        # GNN
        self.W_gnn = nn.ModuleList([nn.Linear(atom_dim, atom_dim),
                                    nn.Linear(atom_dim, atom_dim),
                                    nn.Linear(atom_dim, atom_dim)])
        self.W_gnn_trans = nn.Linear(atom_dim, hid_dim)

        # output
        self.out = nn.Sequential(
            nn.Linear(hid_dim * 4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

        self.do = nn.Dropout(dropout)
        # GAT
        self.atom_dim = atom_dim
        self.compound_attn = nn.ParameterList([nn.Parameter(torch.randn(size=(2 * atom_dim, 1))) for _ in range(len(self.W_gnn))])

    def gnn(self, xs, A):
        for i in range(len(self.W_gnn)):
            h = torch.relu(self.W_gnn[i](xs))
            size = h.size()[0]
            N = h.size()[1]
            a_input = torch.cat([h.repeat(1, 1, N).view(size, N * N, -1), h.repeat(1, N, 1)], dim=2).view(size, N, -1,
                                                                                                          2 * self.atom_dim)
            e = F.leaky_relu(torch.matmul(a_input, self.compound_attn[i]).squeeze(3))
            zero_vec = -9e15 * torch.ones_like(e)
            attention = torch.where(A > 0, e, zero_vec)  # 保证softmax 不为 0
            attention = F.softmax(attention, dim=2)
            attention = F.dropout(attention, self.dropout, training=self.training)
            h_prime = torch.matmul(attention, h)
            xs = xs + h_prime
        xs = self.do(F.relu(self.W_gnn_trans(xs)))
        return xs

    def make_masks(self, atom_num, protein_num, compound_max_len, protein_max_len):
        N = len(atom_num)  # batch size
        compound_mask = torch.zeros((N, compound_max_len))
        protein_mask = torch.zeros((N, protein_max_len))
        for i in range(N):
            compound_mask[i, :atom_num[i]] = 1
            protein_mask[i, :protein_num[i]] = 1
        compound_mask = compound_mask.unsqueeze(1).unsqueeze(3).to(self.device)
        protein_mask = protein_mask.unsqueeze(1).unsqueeze(2).to(self.device)
        return compound_mask, protein_mask

    # compound = [bs,atom_num, atom_dim]
    # adj = [bs, atom_num, atom_num]
    # protein = [bs, protein_len, 100]
    # smi_ids = [bs, smi_len]
    # prot_ids = [bs, prot_len]
    def forward(self, compound, adj, protein, smi_ids, prot_ids, smi_num, prot_num):
        cmp_gnn_out = self.gnn(compound, adj)   # [bs, new_len, hid_dim]
        pro_textcnn_out = self.prot_textcnn(protein) # [bs, prot_len, hid_dim]

        smi_max_len = smi_ids.shape[1]
        prot_max_len = prot_ids.shape[1]

        smi_mask, prot_mask = self.make_masks(smi_num, prot_num, smi_max_len, prot_max_len)
        out_enc_prot = self.enc_prot(self.prot_embed(prot_ids)) #[bs, prot_len, hid_dim]
        out_dec_smi = self.dec_smi(self.smi_embed(smi_ids), out_enc_prot, smi_mask, prot_mask)  # [bs, smi_len, hid_dim]

        prot_mask, smi_mask = self.make_masks(prot_num, smi_num, prot_max_len, smi_max_len)
        out_enc_smi = self.enc_smi(self.smi_embed(smi_ids))  # [bs, smi_len, hid_dim]
        out_dec_prot = self.dec_prot(self.prot_embed(prot_ids), out_enc_smi, prot_mask, smi_mask) # # [bs, prot_len, hid_dim]

        # print(cmp_gnn_out.shape, pro_textcnn_out.shape, out_dec_smi.shape, out_dec_prot.shape)
        is_max = False
        if is_max:
            cmp_gnn_out = cmp_gnn_out.max(dim=1)[0]
            if pro_textcnn_out.ndim>=3: pro_textcnn_out = pro_textcnn_out.max(dim=1)[0]
            out_dec_smi = out_dec_smi.max(dim=1)[0]
            out_dec_prot = out_dec_prot.max(dim=1)[0]
        else:
            cmp_gnn_out = cmp_gnn_out.mean(dim=1)
            if pro_textcnn_out.ndim>=3: pro_textcnn_out = pro_textcnn_out.mean(dim=1)
            out_dec_smi = out_dec_smi.mean(dim=1)
            out_dec_prot = out_dec_prot.mean(dim=1)
        out_fc = torch.cat([cmp_gnn_out, pro_textcnn_out, out_dec_smi, out_dec_prot], dim=-1)
        # print(out_fc.shape)
        return self.out(out_fc)

    def __call__(self, data, train=True):
        compound, adj, protein, correct_interaction, smi_ids, prot_ids, atom_num, protein_num = data
        Loss = nn.CrossEntropyLoss()

        if train:
            predicted_interaction = self.forward(compound, adj, protein, smi_ids, prot_ids, atom_num, protein_num)
            loss = Loss(predicted_interaction, correct_interaction)
            return loss
        else:
            predicted_interaction = self.forward(compound, adj, protein, smi_ids, prot_ids, atom_num, protein_num)
            correct_labels = correct_interaction.to('cpu').data.numpy()
            ys = F.softmax(predicted_interaction, 1).to('cpu').data.numpy()
            predicted_labels = np.argmax(ys, axis=1)
            predicted_scores = ys[:, 1]
            return correct_labels, predicted_labels, predicted_scores

MAX_PROTEIN_LEN = 1500
MAX_DRUG_LEN = 200
def pack(atoms, adjs, proteins, labels, smi_ids, prot_ids, device):
    atoms_len = 0
    proteins_len = 0
    N = len(atoms)
    atom_num, protein_num = [], []

    for atom in atoms:
        if atom.shape[0] >= atoms_len:
            atoms_len = atom.shape[0]

    for protein in proteins:
        if protein.shape[0] >= proteins_len:
            proteins_len = protein.shape[0]

    if atoms_len>MAX_DRUG_LEN: atoms_len = MAX_DRUG_LEN
    atoms_new = torch.zeros((N,atoms_len,34), device=device)
    i = 0
    for atom in atoms:
        a_len = atom.shape[0]
        if a_len>atoms_len: a_len = atoms_len
        atoms_new[i, :a_len, :] = atom[:a_len, :]
        i += 1
    adjs_new = torch.zeros((N, atoms_len, atoms_len), device=device)
    i = 0
    for adj in adjs:
        a_len = adj.shape[0]
        adj = adj + torch.eye(a_len)
        if a_len>atoms_len: a_len = atoms_len
        adjs_new[i, :a_len, :a_len] = adj[:a_len, :a_len]
        i += 1

    if proteins_len>MAX_PROTEIN_LEN: proteins_len = MAX_PROTEIN_LEN
    proteins_new = torch.zeros((N, proteins_len, 100), device=device)
    i = 0
    for protein in proteins:
        a_len = protein.shape[0]
        if a_len>proteins_len: a_len = proteins_len
        proteins_new[i, :a_len, :] = protein[:a_len, :]
        i += 1
    labels_new = torch.zeros(N, dtype=torch.long, device=device)
    i = 0
    for label in labels:
        labels_new[i] = label
        i += 1

    smi_id_len = 0
    for smi_id in smi_ids:
        atom_num.append(len(smi_id))
        if len(smi_id) >= smi_id_len:
            smi_id_len = len(smi_id)

    if smi_id_len>MAX_DRUG_LEN: smi_id_len = MAX_DRUG_LEN
    smi_ids_new = torch.zeros([N, smi_id_len], dtype=torch.long, device=device)
    for i, smi_id in enumerate(smi_ids):
        t_len = len(smi_id)
        if t_len>smi_id_len: t_len = smi_id_len
        smi_ids_new[i, :t_len] = smi_id[:t_len]
    ##########################################################
    prot_id_len = 0
    for prot_id in prot_ids:
        protein_num.append(len(prot_id))
        if len(prot_id) >= prot_id_len: prot_id_len = len(prot_id)

    if prot_id_len>MAX_PROTEIN_LEN: prot_id_len = MAX_PROTEIN_LEN
    prot_ids_new = torch.zeros([N, prot_id_len], dtype=torch.long, device=device)
    for i, prot_id in enumerate(prot_ids):
        t_len = len(prot_id)
        if t_len>prot_id_len: t_len = prot_id_len
        prot_ids_new[i, :t_len] = prot_id[:t_len]
    return (atoms_new, adjs_new, proteins_new, labels_new, smi_ids_new, prot_ids_new, atom_num, protein_num)

from transformers import AdamW
class Trainer(object):
    def __init__(self, model, lr, weight_decay, batch):
        self.model = model
        weight_p, bias_p = [], []
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        for name, p in self.model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
        self.optimizer = AdamW([{'params': weight_p, 'weight_decay': weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=lr)
        self.optimizer = Lookahead(self.optimizer, k=5, alpha=0.5)
        self.batch = batch

    def train(self, dataset, device):
        self.model.train()
        np.random.shuffle(dataset)
        N = len(dataset)
        loss_total = 0
        i = 0
        self.optimizer.zero_grad()
        adjs, atoms, proteins, labels, smi_ids, prot_ids = [], [], [], [], [], []
        for data in dataset:
            i = i+1
            atom, adj, protein, label, smi_id, prot_id = data
            adjs.append(adj)
            atoms.append(atom)
            proteins.append(protein)
            labels.append(label)
            smi_ids.append(smi_id)
            prot_ids.append(prot_id)
            if i % 8 == 0:
                data_pack = pack(atoms, adjs, proteins, labels, smi_ids, prot_ids, device)
                loss = self.model(data_pack)
                # loss = loss / self.batch
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                adjs, atoms, proteins, labels, smi_ids, prot_ids = [], [], [], [], [], []
            else:
                continue
            if i % self.batch == 0 or i == N:
                self.optimizer.step()
                self.optimizer.zero_grad()
            loss_total += loss.item()
        return loss_total

class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataset):
        self.model.eval()
        N = len(dataset)
        T, Y, S = [], [], []
        with torch.no_grad():
            for data in dataset:
                adjs, atoms, proteins, labels, smi_ids, prot_ids = [], [], [], [], [], []
                atom, adj, protein, label, smi_id, prot_id = data
                adjs.append(adj)
                atoms.append(atom)
                proteins.append(protein)
                labels.append(label)
                smi_ids.append(smi_id)
                prot_ids.append(prot_id)

                data = pack(atoms,adjs,proteins, labels, smi_ids, prot_ids, self.model.device)
                correct_labels, predicted_labels, predicted_scores = self.model(data, train=False)
                T.extend(correct_labels)
                Y.extend(predicted_labels)
                S.extend(predicted_scores)
        AUC = roc_auc_score(T, S)
        tpr, fpr, _ = precision_recall_curve(T, S)
        PRC = auc(fpr, tpr)
        precision = precision_score(T, Y)
        recall = recall_score(T, Y)
        return AUC, PRC, precision, recall

    def save_AUCs(self, AUCs, filename):
        with open(filename, 'a') as f:
            f.write('\t'.join(map(str, AUCs)) + '\n')

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)
