"""
HADM: Hypergraph-guided Association Discrepancy Mining
多变量时间序列异常检测
"""

import os
import argparse
import random
import numpy as np
import torch
from torch.backends import cudnn
from solver import HADMSolver


def set_seed(seed=42):
    """固定所有随机种子，保证可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(
        description='HADM: Hypergraph-guided Association Discrepancy Mining')

    # ---- 运行模式 ----
    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'test'],
                        help='train 或 test')
    parser.add_argument('--dataset', type=str, default='PSM',
                        help='数据集名称 (PSM, SMAP, MSL, SMD, SWaT)')
    parser.add_argument('--data_path', type=str, default='./dataset/PSM/',
                        help='数据集根目录')
    parser.add_argument('--model_save_path', type=str, default='checkpoints',
                        help='模型与结果保存路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子（用于多种子实验）')

    # ---- 训练参数 ----
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=256, help='批次大小')
    parser.add_argument('--patience', type=int, default=3, help='早停耐心值')
    parser.add_argument('--anormly_ratio', type=float, default=-1.0,
                        help='阈值标定的异常比例（百分位数），-1=按数据集自动选择')

    # ---- 序列参数 ----
    parser.add_argument('--win_size', type=int, default=100, help='窗口长度 L')
    parser.add_argument('--input_c', type=int, default=25, help='传感器数量 C')

    # ---- 时间分支参数（继承自 Anomaly Transformer）----
    parser.add_argument('--d_model', type=int, default=512, help='模型维度')
    parser.add_argument('--n_heads', type=int, default=8, help='注意力头数')
    parser.add_argument('--e_layers', type=int, default=3, help='编码器层数')
    parser.add_argument('--d_ff', type=int, default=512, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 率')

    # ---- 关系分支参数（本文贡献）----
    parser.add_argument('--d_h', type=int, default=16, help='超图隐层维度')
    parser.add_argument('--k_hypergraph', type=int, default=3,
                        help='超图 Top-K 邻居数')

    # ---- 训练损失系数 ----
    parser.add_argument('--k', type=float, default=0.5,
                        help='时间关联差异系数（极小极大目标权重）')
    parser.add_argument('--lambda_r', type=float, default=0.05,
                        help='关系关联差异系数 λ_r（Phase 1 权重）')

    # ---- 门控融合参数 ----
    parser.add_argument('--beta', type=float, default=0.0,
                        help='融合偏置（0=纯数据驱动，>0=偏向时间分支）')

    args = parser.parse_args()
    config = vars(args)
    config['k_hypergraph'] = max(1, int(config['k_hypergraph']))

    # 按数据集自动选择 anomaly ratio（与 Anomaly Transformer 一致）
    if config['anormly_ratio'] < 0:
        ratio_map = {'SWAT': 0.1, 'SMD': 0.5}
        config['anormly_ratio'] = ratio_map.get(config['dataset'], 1.0)

    # 固定随机种子
    set_seed(args.seed)

    # 打印配置
    print('=' * 55)
    print(f"HADM  |  模式={args.mode}  数据集={args.dataset}  seed={args.seed}")
    print('=' * 55)
    for key in sorted(config):
        print(f"  {key:<22s}: {config[key]}")
    print('=' * 55)

    os.makedirs(args.model_save_path, exist_ok=True)

    solver = HADMSolver(config)
    if args.mode == 'train':
        solver.train()
        print(f"\n训练完成，模型已保存至：{args.model_save_path}")
        print("使用 --mode test 进行评估。")
    else:
        solver.test()


if __name__ == '__main__':
    main()
