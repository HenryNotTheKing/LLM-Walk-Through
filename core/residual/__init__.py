"""残差连接与跨层信息流：AttnRes、mHC 等拓扑变体。"""

from core.residual.attn_res import (
    AttnResAggregator,
    AttnResState,
    PreNormBlockWithAttnRes,
    block_attn_res,
    finalize_attn_res_block,
    init_attn_res_state,
    update_attn_res_partial,
)
from core.residual.mhc import (
    ManifoldHyperConnections,
    PreNormBlockWithMHC,
    collapse_streams,
    expand_to_streams,
    sinkhorn_knopp,
)

__all__ = [
    "AttnResAggregator",
    "AttnResState",
    "PreNormBlockWithAttnRes",
    "ManifoldHyperConnections",
    "PreNormBlockWithMHC",
    "block_attn_res",
    "collapse_streams",
    "expand_to_streams",
    "finalize_attn_res_block",
    "init_attn_res_state",
    "sinkhorn_knopp",
    "update_attn_res_partial",
]
