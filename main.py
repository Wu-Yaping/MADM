"""
MADM: Multivariate Association Discrepancy Mining
for Multivariate Time Series Anomaly Detection
"""

import os
import argparse
from torch.backends import cudnn
from solver import MADMSolver


def main():
    parser = argparse.ArgumentParser(
        description='MADM: Multivariate Association Discrepancy Mining')

    # ---- Mode ----
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='train or test')
    parser.add_argument('--dataset', type=str, default='PSM',
                        help='dataset name (PSM, SMAP, MSL, SMD, SWaT)')
    parser.add_argument('--data_path', type=str, default='./dataset/PSM/',
                        help='root directory of the dataset')
    parser.add_argument('--model_save_path', type=str, default='checkpoints',
                        help='path to save models and results')

    # ---- Training ----
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--anormly_ratio', type=float, default=4.0,
                        help='anomaly ratio for threshold calibration (percentile)')

    # ---- Sequence ----
    parser.add_argument('--win_size', type=int, default=100, help='window length L')
    parser.add_argument('--input_c', type=int, default=25, help='number of sensors C')

    # ---- Temporal branch (inherited from Anomaly Transformer) ----
    parser.add_argument('--d_model', type=int, default=512, help='model dimension')
    parser.add_argument('--n_heads', type=int, default=8, help='number of attention heads')
    parser.add_argument('--e_layers', type=int, default=3, help='number of encoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='feed-forward dimension')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout rate')

    # ---- Spatial branch (proposed contribution) ----
    parser.add_argument('--d_h', type=int, default=16, help='hypergraph hidden dimension')
    parser.add_argument('--k_hypergraph', type=int, default=3,
                        help='Top-K neighbors for hypergraph construction')

    # ---- Loss coefficients ----
    parser.add_argument('--k', type=float, default=0.5,
                        help='temporal discrepancy coefficient (minimax weight)')
    parser.add_argument('--lambda_r', type=float, default=0.05,
                        help='spatial discrepancy coefficient lambda_r (Phase 1 weight)')

    # ---- Gated fusion ----
    parser.add_argument('--beta', type=float, default=0.0,
                        help='fusion bias (0=data-driven, >0=biased toward temporal branch)')

    args = parser.parse_args()
    config = vars(args)
    config['k_hypergraph'] = max(1, int(config['k_hypergraph']))

    print('=' * 55)
    print(f"MADM  |  mode={args.mode}  dataset={args.dataset}")
    print('=' * 55)
    for key in sorted(config):
        print(f"  {key:<22s}: {config[key]}")
    print('=' * 55)

    cudnn.benchmark = True
    os.makedirs(args.model_save_path, exist_ok=True)

    solver = MADMSolver(config)
    if args.mode == 'train':
        solver.train()
        print(f"\nTraining complete. Model saved to: {args.model_save_path}")
        print("Run with --mode test to evaluate.")
    else:
        solver.test()


if __name__ == '__main__':
    main()
