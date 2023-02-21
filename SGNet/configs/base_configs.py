import argparse

__all__ = ['parse_base_args']


def parse_base_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='', type=str)
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--weight_decay', default=5e-04, type=float)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    parser.add_argument('--shuffle', default=True, type=bool)
    parser.add_argument('--age', default='all',
                        type=str)  # Options: child, adult, elderly, no_label, labeled (only for JAAD), all
    parser.add_argument('--split', default='test', type=str)  # Options: train, val, test
    parser.add_argument('--gender', default='all', type=str)  # Options: all, male, female
    return parser
