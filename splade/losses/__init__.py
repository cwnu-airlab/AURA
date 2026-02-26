# from .contrastive import ClsInfoNCELoss, SparseClsCombinedLoss
from .pairwise import (DistilKLLoss, DistilMarginMSE, InBatchPairwiseNLL, InBatchPairwiseNLLWithCls,
                       PairwiseBPR, PairwiseNLL)
from .pointwise import BCEWithLogitsLoss
