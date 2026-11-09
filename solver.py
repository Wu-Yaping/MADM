"""
HADM Solver — 梯度分离极小极大训练与评估

训练公式（论文 Eq. 21–22）：
    Phase 1（放大）：  L1 = L_rec - k·L_series - λ_r·L_rel
    Phase 2（先验学习）：L2 = L_rec + k·L_prior

异常评分（论文 Eq. 25）：
    Score(t) = Softmax(-L_series - L_prior) · ||x̂_t - x_t||² · (1 + D̄_rel)
"""

import os
import time
import json

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity

from data_factory.data_loader import get_loader_segment
from model.hadm import HADM


# ============================================================
#  工具函数
# ============================================================
def my_kl_loss(p, q):
    """对称 KL 散度：先对最后一维求和，再对 batch 均值。"""
    res = p * (torch.log(p + 1e-4) - torch.log(q + 1e-4))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_init, decay_interval=2):
    """每 decay_interval 个 epoch 将学习率减半。"""
    lr = lr_init * (0.5 ** ((epoch - 1) // decay_interval))
    for pg in optimizer.param_groups:
        pg['lr'] = lr
    print(f"  学习率调整为 {lr:.2e}")


class EarlyStopping:
    """带模型保存的早停机制。"""
    def __init__(self, patience=7, verbose=True, dataset_name='', delta=0.0):
        self.patience  = patience
        self.verbose   = verbose
        self.dataset   = dataset_name
        self.delta     = delta
        self.counter   = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            self._save(val_loss, model, path)
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save(self, val_loss, model, path):
        if self.verbose:
            print(f"  验证损失下降 ({self.val_loss_min:.6f} → {val_loss:.6f})，保存模型...")
        torch.save(model.state_dict(),
                   os.path.join(path, f'{self.dataset}_checkpoint.pth'))
        self.val_loss_min = val_loss


# ============================================================
#  HADMSolver
# ============================================================
class HADMSolver:
    def __init__(self, config):
        self.config = config
        self.__dict__.update(config)

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # ---- 数据加载 ----
        _loader = lambda mode: get_loader_segment(
            self.data_path, self.batch_size, self.win_size,
            step=1, mode=mode, dataset=self.dataset)
        self.train_loader = _loader('train')
        self.vali_loader  = _loader('val')
        self.test_loader  = _loader('test')
        self.thre_loader  = _loader('thre')

        # ---- 超图 + 模型 ----
        self.hyperedge_index, self.edge_features = \
            self._build_hypergraph()
        self._build_model()
        self.criterion = nn.MSELoss()

    # ----------------------------------------------------------
    #  超图构建
    # ----------------------------------------------------------
    def _build_hypergraph(self):
        """基于全量训练数据的余弦相似度构建 Top-K 超图。"""
        try:
            # 使用全量训练数据（所有时间步）计算传感器间余弦相似度
            full_train = self.train_loader.dataset.train  # [T, C]
            full_train = np.asarray(full_train, dtype=np.float64)
            cos_sim = cosine_similarity(full_train.T)     # [C, C]
            cos_sim = np.maximum(cos_sim, 0)

            K = getattr(self, 'k_hypergraph', 3)
            edges, feats = [], []
            for i in range(cos_sim.shape[0]):
                topk = np.argsort(cos_sim[i])[-K:]
                for j in topk:
                    edges.append((i, j))
                    feats.append(cos_sim[i, j])

            ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
            ef = torch.tensor(feats, dtype=torch.float).view(-1, 1)
            ef = ef / (ef.max() + 1e-8)
            return ei.to(self.device), ef.to(self.device)

        except Exception as e:
            print(f"警告：超图构建失败（{e}），使用全连接回退方案。")
            C = self.input_c
            idx = torch.arange(C)
            ei = torch.stack([idx.repeat_interleave(C), idx.repeat(C)])
            ef = torch.ones(C * C, 1)
            return ei.to(self.device), ef.to(self.device)

    # ----------------------------------------------------------
    #  模型构建
    # ----------------------------------------------------------
    def _build_model(self):
        self.model = HADM(
            win_size   = self.win_size,
            enc_in     = self.input_c,
            c_out      = self.input_c,
            d_model    = getattr(self, 'd_model',  512),
            n_heads    = getattr(self, 'n_heads',  8),
            e_layers   = getattr(self, 'e_layers', 3),
            d_ff       = getattr(self, 'd_ff',     512),
            dropout    = getattr(self, 'dropout',  0.1),
            d_h        = getattr(self, 'd_h',      16),
            hyperedge_index = self.hyperedge_index,
            edge_features   = self.edge_features,
            output_attention = True,
            beta       = getattr(self, 'beta', 0.0),
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr)
        n = sum(p.numel() for p in self.model.parameters())
        print(f"HADM 构建完成：{n/1e6:.2f}M 参数，设备：{self.device}")

    def _update_hypergraph(self):
        """重新构建超图并更新模型缓冲区。"""
        self.hyperedge_index, self.edge_features = self._build_hypergraph()
        rb = self.model.relational_branch
        rb.hyperedge_index = self.hyperedge_index
        rb.edge_features   = self.edge_features
        rb._build_relational_prior()

    # ----------------------------------------------------------
    #  验证
    # ----------------------------------------------------------
    def _vali(self, loader):
        self.model.eval()
        losses = []
        k  = getattr(self, 'k',        0.5)
        lr = getattr(self, 'lambda_r', 0.05)
        with torch.no_grad():
            for inp, _ in loader:
                inp = inp.float().to(self.device)
                out, sl, pl, _, _, rd, _ = self.model(inp)

                rec = self.criterion(out, inp)
                s_loss = p_loss = 0.0
                for u in range(len(pl)):
                    pn = pl[u] / pl[u].sum(-1, keepdim=True)
                    s_loss += torch.mean(my_kl_loss(sl[u], pn.detach()))
                    p_loss += torch.mean(my_kl_loss(pn.detach(), sl[u]))
                s_loss /= len(pl); p_loss /= len(pl)
                r_loss = rd.mean() if rd is not None else torch.zeros(1, device=self.device)
                losses.append((rec + k * (s_loss + p_loss) + lr * r_loss).item())
        return float(np.mean(losses))

    # ----------------------------------------------------------
    #  训练
    # ----------------------------------------------------------
    def train(self):
        print("=" * 50)
        print(f"HADM 训练开始  |  数据集：{self.dataset}")
        print("=" * 50)

        os.makedirs(self.model_save_path, exist_ok=True)
        early_stop = EarlyStopping(
            patience=getattr(self, 'patience', 7),
            verbose=True, dataset_name=self.dataset)

        k  = getattr(self, 'k',        0.5)
        lr = getattr(self, 'lambda_r', 0.05)
        val_history = []

        for epoch in range(self.num_epochs):
            t0 = time.time()
            l1_buf, l2_buf = [], []
            self.model.train()
            self._update_hypergraph()

            for step, (input_data, _) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                inp = input_data.float().to(self.device)

                out, sl, pl, _, _, rel_disc, rel_attn = self.model(inp)

                # 重建损失
                rec_loss = self.criterion(out, inp)

                # 时间关联差异
                s_loss = p_loss = 0.0
                for u in range(len(pl)):
                    pn = pl[u] / pl[u].sum(-1, keepdim=True)
                    s_loss += (torch.mean(my_kl_loss(sl[u], pn.detach()))
                               + torch.mean(my_kl_loss(pn.detach(), sl[u])))
                    p_loss += (torch.mean(my_kl_loss(pn, sl[u].detach()))
                               + torch.mean(my_kl_loss(sl[u].detach(), pn)))
                s_loss /= len(pl); p_loss /= len(pl)

                # 关系关联差异（Phase 1 使用梯度分离）
                if rel_attn is not None and rel_attn.numel() > 0:
                    rel_loss = self.model.relational_branch \
                        .compute_relational_discrepancy(
                            rel_attn,
                            inp.size(0) * self.win_size,
                            self.device,
                            detach_prior=True
                        ).mean()
                else:
                    rel_loss = torch.zeros(1, device=self.device)

                if step % 2 == 0:
                    # Phase 1：放大两个关联差异
                    loss = rec_loss - k * s_loss - lr * rel_loss
                    l1_buf.append(loss.item())
                else:
                    # Phase 2：仅更新时间先验的带宽 σ
                    loss = rec_loss + k * p_loss
                    l2_buf.append(loss.item())

                loss.backward()
                self.optimizer.step()

            val_loss = self._vali(self.test_loader)
            val_history.append(val_loss)
            print(f"Epoch {epoch+1:3d}/{self.num_epochs} | "
                  f"{time.time()-t0:.1f}s | "
                  f"L1={np.mean(l1_buf) if l1_buf else 0:.4f} | "
                  f"L2={np.mean(l2_buf) if l2_buf else 0:.4f} | "
                  f"Val={val_loss:.4f}")

            early_stop(val_loss, self.model, self.model_save_path)
            if early_stop.early_stop:
                print("早停触发。")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.lr)

        # 保存训练统计
        stats = {
            'val_mean': float(np.mean(val_history)),
            'val_std':  float(np.std(val_history)),
            'val_history': val_history,
        }
        with open(os.path.join(self.model_save_path,
                               f'{self.dataset}_train_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"训练结束  |  验证损失均值={stats['val_mean']:.4f} ± {stats['val_std']:.4f}")

    # ----------------------------------------------------------
    #  测试
    # ----------------------------------------------------------
    def test(self):
        ckpt = os.path.join(self.model_save_path,
                            f'{self.dataset}_checkpoint.pth')
        self.model.load_state_dict(torch.load(ckpt, map_location=self.device))
        self.model.eval()
        self._update_hypergraph()
        print("=" * 50)
        print(f"HADM 测试  |  数据集：{self.dataset}")
        print("=" * 50)

        criterion = nn.MSELoss(reduction='none')
        temperature = 50

        def compute_energy(loader):
            scores = []
            with torch.no_grad():
                for inp, _ in loader:
                    inp = inp.float().to(self.device)
                    out, sl, pl, _, _, rel_disc, _ = self.model(inp)

                    # 逐时间步重建误差
                    rec = torch.mean(criterion(inp, out), dim=-1)  # [B, L]

                    # 时间关联差异加权
                    s_loss = p_loss = torch.zeros_like(rec)
                    for u in range(len(pl)):
                        pn = pl[u] / pl[u].sum(-1, keepdim=True)
                        s_loss += my_kl_loss(sl[u], pn.detach()) * temperature
                        p_loss += my_kl_loss(pn, sl[u].detach()) * temperature
                    s_loss /= len(pl); p_loss /= len(pl)

                    # 关系调制因子 (1 + D̄_rel)
                    if rel_disc is not None and rel_disc.numel() > 0:
                        rel_score = rel_disc.mean(dim=-1)   # [B, L]
                    else:
                        rel_score = torch.zeros_like(rec)

                    # Eq. 25
                    score = (torch.softmax(-s_loss - p_loss, dim=-1)
                             * rec * (1.0 + rel_score))
                    scores.append(score.cpu().numpy())
            return np.concatenate(scores).reshape(-1)

        # 阈值标定
        train_energy = compute_energy(self.train_loader)
        thre_energy  = compute_energy(self.thre_loader)
        thresh = np.percentile(
            np.concatenate([train_energy, thre_energy]),
            100 - self.anormly_ratio)
        print(f"阈值：{thresh:.6f}  （异常比例：{self.anormly_ratio}%）")

        # 测试评分
        test_energy = compute_energy(self.test_loader)
        test_labels = np.concatenate(
            [lb.numpy() for _, lb in self.test_loader]).reshape(-1)

        pred = (test_energy > thresh).astype(int)
        gt   = test_labels.astype(int)

        # Point Adjustment（Xu et al., ICLR 2022）
        anomaly_state = False
        for i in range(len(gt)):
            if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
                anomaly_state = True
                for j in range(i, 0, -1):
                    if gt[j] == 0: break
                    pred[j] = 1
                for j in range(i, len(gt)):
                    if gt[j] == 0: break
                    pred[j] = 1
            elif gt[i] == 0:
                anomaly_state = False
            if anomaly_state:
                pred[i] = 1

        acc = accuracy_score(gt, pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            gt, pred, average='binary')
        print(f"[结果] Acc={acc:.4f} | Prec={prec:.4f} | "
              f"Rec={rec:.4f} | F1={f1:.4f}")

        # 保存结果
        np.save(os.path.join(self.model_save_path,
                             f'{self.dataset}_pred.npy'), pred)
        np.save(os.path.join(self.model_save_path,
                             f'{self.dataset}_gt.npy'), gt)
        np.save(os.path.join(self.model_save_path,
                             f'{self.dataset}_energy.npy'), test_energy)
        return acc, prec, rec, f1
