
import argparse

import numpy as np

import random
import torch
import torch.nn.functional as F
import dgl
from dgl import logging
from imblearn.metrics import geometric_mean_score
from sklearn.metrics import f1_score, roc_auc_score

from model import GAT_COBO


from dgl.data.utils import load_graphs
from dgl.data.utils import makedirs, save_info, load_info
from sklearn.model_selection import train_test_split

import os

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.seed(seed)
    dgl.random.seed(seed)

def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits,_, attention = model(features)
        logits = torch.reshape(logits, [logits.shape[0], -1])
        logits = logits[mask]
        labels = labels[mask]
        loss_fcn = torch.nn.CrossEntropyLoss()
        loss = loss_fcn(logits, labels)
        return accuracy(logits, labels), loss, logits


def gen_mask(g, train_rate, val_rate, IR, IR_set):
    labels = g.ndata['label']
    g.ndata['label'] = labels.long()
    labels = np.array(labels)
    n_nodes = len(labels)
    index = list(range(n_nodes))  # 将所有节点都视为测试节点
    train_idx, val_idx, _, _ = train_test_split(index, labels, stratify=labels, train_size=train_rate, test_size=val_rate,
                                                 random_state=2, shuffle=True)
    train_mask = torch.zeros(n_nodes, dtype=torch.bool)
    val_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask = torch.zeros(n_nodes, dtype=torch.bool)
    test_mask[index] = True  # 将所有节点都设置为测试节点
    g.ndata['train_mask'] = train_mask
    g.ndata['val_mask'] = val_mask
    g.ndata['test_mask'] = test_mask
    return g, train_idx

# 加载模型
parser = argparse.ArgumentParser(description='GAT-COBO')
parser.add_argument('--dataset', type=str, default='Sichuan', help='Sichuan,BUPT')
parser.add_argument("--dropout", type=float, default=0.3, help="dropout probability")
parser.add_argument("--adj_dropout", type=float, default=0.3, help="mixed dropout for adj")
parser.add_argument('--layers', type=int, default=8, help='Number of Basic-model layers.')
parser.add_argument("--num_layers", type=int, default=1, help="number of attention-hidden layers")
parser.add_argument('--hid', type=int, default=64, help='Number of hidden units. ')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for optimizer(all layers).')
parser.add_argument('--reg', type=float, default=5e-3, help='Weight decay on the 1st layer.')
parser.add_argument("--epochs", type=int, default=400, help="number of training epochs")
parser.add_argument('--patience', type=int, default=200, help='patience in early stopping')
parser.add_argument('--num_heads', type=int, default=1, help='number of hidden attention heads')
parser.add_argument("--num_out_heads", type=int, default=1, help="number of output attention heads")
parser.add_argument("--in_drop", type=float, default=0.1, help="input feature dropout")
parser.add_argument("--attn_drop", type=float, default=0.1, help="attention dropout")
parser.add_argument('--early_stop', action='store_true', default=False,
                    help="indicates whether to use early stop or not")
parser.add_argument("--residual", action="store_true", default=False, help="use residual connection")
parser.add_argument('--negative_slope', type=float, default=0.2, help="the negative slope of leaky relu")
parser.add_argument('--print_interval', type=int, default=50, help="the interval of printing in training")
parser.add_argument('--seed', type=int, default=42, help="seed for our system")
parser.add_argument('--att_loss_weight', type=float, default=0.5, help="attention loss weight")
parser.add_argument('--attention_weight', type=float, default=0.7, help='External Attention coefficient.')
parser.add_argument('--feature_weight', type=float, default=0.4, help='Feature adjust coefficient about attention.')
parser.add_argument('--train_size', type=float, default=0.2, help='train size.')
parser.add_argument('--blank', type=int, default=0, help='use during find best hyperparameter.')
parser.add_argument('--IR', type=float, default=0.1, help='imbalanced ratio.')
parser.add_argument('--IR_set', type=int, default=0, help='whether to set imbalanced ratio,1 for set ,0 for not.')
parser.add_argument('--cost', type=int, default=2, help="set the way to calculate cost matrix,0:'uniform',1:'inverse',2:'log1p-inverse' ")
args = parser.parse_args()
#print(args)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


setup_seed(args.seed)
if args.dataset == 'Sichuan':
    dataset, _ = load_graphs("./data/Sichuan_tele_test.bin")
    n_classes = load_info("./data/Sichuan_tele_test.pkl")['num_classes']
    graph = dataset[0]
    g,train_idx = gen_mask(graph, args.train_size, 0.7,args.IR,args.IR_set)
for e in g.etypes:
    g = g.int().to(device)
    dgl.remove_self_loop(g,etype=e)
    dgl.add_self_loop(g,etype=e)
features = g.ndata['feat'].float()
labels = g.ndata['label']
train_mask = g.ndata['train_mask']

val_mask = g.ndata['val_mask']
test_mask = g.ndata['test_mask']
num_feats = features.shape[1]
num_edges = g.num_edges()
features = g.ndata['feat'].float()
heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
model = GAT_COBO(g,
                args.num_layers,
                num_feats,
                args.hid,
                n_classes,
                heads,
                F.elu,
                args.dropout,
                args.adj_dropout,
                args.in_drop,
                args.attn_drop,
                args.negative_slope,
                args.residual
                )
model.load_state_dict(torch.load('F:\dataset\model.pth'))
print("模型加载成功")
model.to(device)
# 设置模型为评估模式
model.eval()
results = torch.zeros(g.adj().shape[0], n_classes).to(device)


# 进行预测
with torch.no_grad():
    test_acc, test_loss, test_logits = evaluate(model, features, labels, test_mask)
    anomaly_indicator = torch.sigmoid(test_logits)

    test_h = torch.argmax(test_logits, dim=1)
    test_acc = torch.sum(test_h == labels[test_mask]) * 1.0 / len(labels[test_mask])
    test_f1 = f1_score(labels[test_mask].cpu(), test_h.cpu(), average='weighted')
    test_auc = roc_auc_score(labels[test_mask].cpu(), test_h.cpu(), average='macro',
                             multi_class='ovo')
    print( f"测试集 准确率: {test_acc * 100:.1f}%  测试集 F1-score: {test_f1 * 100:.1f}%  测试集 AUC: {test_auc * 100:.1f}%")

    #print(len(labels[test_mask]))
    #print(test_acc)