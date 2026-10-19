from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external/kermut"))
from src.model.kernel import Kermut


class WildTypeKermut(Kermut):
    def forward(self, x1, x2, diag=False, **kwargs):
        assert x1.ndim == 2 and x2.ndim == 2
        tokens1 = x1.reshape(-1, self.seq_len, 20).argmax(dim=-1)
        tokens2 = x2.reshape(-1, self.seq_len, 20).argmax(dim=-1)
        rows = torch.nonzero((tokens1 != self.wt_toks).any(dim=-1)).flatten()
        columns = torch.nonzero((tokens2 != self.wt_toks).any(dim=-1)).flatten()
        # 参考序列的突变集合为空，其结构核协方差严格为零。
        covariance = x1.new_zeros((len(x1), len(x2)))
        if len(rows) and len(columns):
            mutant_covariance = super().forward(x1[rows], x2[columns], **kwargs)
            covariance = covariance.index_put((rows[:, None], columns[None, :]), mutant_covariance)
        if diag:
            assert len(x1) == len(x2)
            return covariance.diagonal()
        return covariance
