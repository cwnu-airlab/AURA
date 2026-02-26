import torch
import torch.nn.functional as F
import math

from .pairwise import (DistilKLLoss, DistilMarginMSE, InBatchPairwiseNLL,
                       PairwiseBPR, PairwiseNLL)


class ClsInfoNCELoss:
    """InfoNCE loss computed on CLS embeddings.

    Expected keys in the trainer output dict:
      - 'pos_q_cls': CLS of positive queries (batch, hidden)
      - 'pos_d_cls': CLS of positive documents (batch, hidden)
      - optionally 'neg_d_cls': CLS of negative documents (batch, hidden)
    """

    def __init__(self, temperature=0.05, normalize=True, include_triplet_negatives=True):
        self.temperature = temperature
        self.normalize = normalize
        self.include_triplet_negatives = include_triplet_negatives
        self.criterion = torch.nn.CrossEntropyLoss()

    def __call__(self, out_d):
        query_cls = out_d["pos_q_cls"]
        pos_doc_cls = out_d["pos_d_cls"]
        neg_doc_cls = out_d.get("neg_d_cls")

        if self.normalize:
            query_cls = F.normalize(query_cls, dim=-1)
            pos_doc_cls = F.normalize(pos_doc_cls, dim=-1)
            if neg_doc_cls is not None:
                neg_doc_cls = F.normalize(neg_doc_cls, dim=-1)

        # in-batch positives act as negatives for other queries
        logits = torch.matmul(query_cls, pos_doc_cls.t())

        if self.include_triplet_negatives and neg_doc_cls is not None:
            logits = torch.cat([logits, torch.matmul(query_cls, neg_doc_cls.t())], dim=1)

        logits = logits / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        loss = self.criterion(logits, targets)
        return loss


class SparseClsCombinedLoss:
    """Combine sparse ranking loss with CLS InfoNCE."""

    _SPARSE_LOSSES = {
        "PairwiseNLL": PairwiseNLL,
        "InBatchPairwiseNLL": InBatchPairwiseNLL,
        "PairwiseBPR": PairwiseBPR,
        "DistilMarginMSE": DistilMarginMSE,
        "KlDiv": DistilKLLoss,
    }
    

    def __init__(self, 
                sparse_loss_name="InBatchPairwiseNLL",
                normalize=True,
                include_triplet_negatives=True):
        try:
            self.sparse_loss = self._SPARSE_LOSSES[sparse_loss_name]()
        except KeyError as exc:
            raise ValueError(f"Unsupported sparse loss '{sparse_loss_name}'") from exc
        
        self.cls_loss = ClsInfoNCELoss(normalize=normalize,
                                       include_triplet_negatives=include_triplet_negatives)

    def __call__(self, out_d):
        sparse_val = self.sparse_loss(out_d)
        cls_val = self.cls_loss(out_d)
        return sparse_val, cls_val
