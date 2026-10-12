import argparse

parser = argparse.ArgumentParser(
    "Space targets with micro-motion recognition by RCS/HRRP/Text")

def get_args():
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--use_cl', type=bool, default=True, help='Whether use loss_cl or not')
    parser.add_argument('--rht_output_dim', type=int, default=128, help='Whether use loss_cl or not')
    parser.add_argument('--gat_hidden_dim', type=int, default=128, help='Whether use loss_cl or not')
    parser.add_argument('--gat_output_dim', type=int, default=128, help='Whether use loss_cl or not')
    parser.add_argument('--num_class', type=int, default=4, help='classification number')
    parser.add_argument('--lr', type=float, default=0.001,help='learning rate')
    parser.add_argument('--betas', default=(0.9, 0.99), help='betas of Adam optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='L2 of optimizer')
    parser.add_argument('--gamma', type=float, default=0.97, help='Learning rate decay rate')
    parser.add_argument('--epochs', type=int, default=50, help='The epoch of training')
    parser.add_argument('--alpha', type=float, default=0.8, help='alpha weights')
    parser.add_argument('--beta', type=float, default=0.2, help='beta weights')
    return parser