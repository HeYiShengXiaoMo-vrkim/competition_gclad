import argparse
import torch
import torch.nn.functional as F
import dgl
import os
from dgl.data.utils import load_graphs, load_info
from model import GAT_COBO

# 简单的单节点(或样本)预测脚本
# 用法示例:
#   python predict_single.py --dataset Sichuan --node_id 10 --model ./models/gat_cobo_model.pth
# 若你想基于 BUPT 则 --dataset BUPT

def load_dataset(name: str):
    if name == 'Sichuan':
        dataset, _ = load_graphs('./data/Sichuan_tele.bin')
        info = load_info('./data/Sichuan_tele.pkl')
    elif name == 'BUPT':
        dataset, _ = load_graphs('./data/BUPT_tele.bin')
        info = load_info('./data/BUPT_tele.pkl')
    else:
        raise ValueError('Unsupported dataset name')
    g = dataset[0]
    return g, info['num_classes']

def build_model(g, n_classes, args):
    num_feats = g.ndata['feat'].shape[1]
    heads = ([args.num_heads] * args.num_layers) + [args.num_out_heads]
    model = GAT_COBO(
        g,
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
    return model

def softmax_prob(logits):
    return F.softmax(logits, dim=-1)

def main():
    parser = argparse.ArgumentParser(description='Single node prediction')
    parser.add_argument('--dataset', type=str, default='Sichuan')
    parser.add_argument('--model', type=str, required=True, help='Path to saved model .pth')
    parser.add_argument('--node_id', type=int, required=True, help='Node index in the graph')
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=1)
    parser.add_argument('--num_out_heads', type=int, default=1)
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--adj_dropout', type=float, default=0.3)
    parser.add_argument('--in_drop', type=float, default=0.1)
    parser.add_argument('--attn_drop', type=float, default=0.1)
    parser.add_argument('--negative_slope', type=float, default=0.2)
    parser.add_argument('--residual', action='store_true', default=False)
    parser.add_argument('--cpu', action='store_true', default=False)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and (not args.cpu) else 'cpu')
    g, n_classes = load_dataset(args.dataset)
    g = g.to(device)
    model = build_model(g, n_classes, args).to(device)

    if not os.path.isfile(args.model):
        raise FileNotFoundError(f'Model file not found: {args.model}')
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()

    with torch.no_grad():
        features = g.ndata['feat'].float().to(device)
        logits_inter, logits_inner, _ = model(features)
        logits = logits_inter.reshape(logits_inter.shape[0], -1)
        if args.node_id < 0 or args.node_id >= logits.shape[0]:
            raise IndexError('node_id out of range')
        node_logits = logits[args.node_id]
        probs = softmax_prob(node_logits)
        pred_label = torch.argmax(probs).item()
        print({
            'node_id': args.node_id,
            'pred_label': pred_label,
            'probs': probs.cpu().numpy().tolist()
        })

if __name__ == '__main__':
    main()
