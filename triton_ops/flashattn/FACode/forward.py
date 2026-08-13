import torch.nn as nn
import triton
import triton.language as tl

configs = [
    triton.Config({'BLOCK_SIZE_M': BM, 'BLOCK_SIZE_N': BN}, num_stages=s, num_warps=w) \
    for BM in [64, 128]\
    for BN in [32, 64, 128]\
    for s in [2, 3, 4, 5] \
    for w in [2, 4, 8]\
]

@triton.jit
def _forward_inner_kv(
    Q,
    k,
    v,
    O_block,
    qk_scale,
    m_i,
    l_i,
    SEQ_LEN,
    HEAD_DIM_Q: tl.constexpr,
    stage: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    num_stages: tl.constexpr,
):
  q_group = tl.program_id(0)
  batch_index_head = tl.program_id(1)

  #set limit for KJ blocks
  diag_start = q_group * BLOCK_SIZE_M // BLOCK_SIZE_N
  diag_block = BLOCK_SIZE_M // BLOCK_SIZE_N

  if stage == 1:
    low, high = 0, (SEQ_LEN + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
  elif stage == 2:
    low, high = diag_start, diag_start + diag_block
  elif stage == 3:
    low, high = 0, diag_start

  offset_head_dim = tl.arange(0, HEAD_DIM_Q)
  offset_N = tl.arange(0, BLOCK_SIZE_N)
  offs_m = q_group * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

  #offsets for kt and j
  offset_kt = (
                offset_head_dim[:, None] +
                offset_N[None, :] * HEAD_DIM_Q
              )
  offset_v = (
                offset_N[:, None] * HEAD_DIM_Q +
                offset_head_dim[None, :]
              )
  
  #load our KV blocks
  for j in tl.range(low, high, 1, num_stages=num_stages):
    #calculate our k and j pointer
    k_ptr = k + batch_index_head * SEQ_LEN * HEAD_DIM_Q + j * BLOCK_SIZE_N * HEAD_DIM_Q
    v_ptr = v + batch_index_head * SEQ_LEN * HEAD_DIM_Q + j * BLOCK_SIZE_N * HEAD_DIM_Q


    #generate mask
    rows = j * BLOCK_SIZE_N + offset_N
    mask_kt = rows[None, :] < SEQ_LEN
    mask_v = rows[:, None] < SEQ_LEN

    #load kv blocks
    kt = tl.load(k_ptr + offset_kt, mask=mask_kt, other=0.0)
    v_block = tl.load(v_ptr + offset_v, mask = mask_v, other=0.0)

    #compute Q@Kt
    QKt = tl.dot(Q, kt)
    QKt *= qk_scale

    # Remove padded key columns from softmax.
    QKt = tl.where(
        rows[None, :] < SEQ_LEN,
        QKt,
        -float("inf"),
    )

    #if stage 2, we apply a mask to our mask to our QKt values
    if stage == 2:
      mask = offs_m[:, None] >= rows[None, :]
      QKt += tl.where(mask, 0, float("-inf"))

    m_ij = tl.maximum(m_i, tl.max(QKt, axis=1))
    QKt -= m_ij[:, None]

    P = tl.math.exp2(QKt)
    correction = tl.math.exp2(m_i - m_ij)

    #correct our previous accumulations
    l_ij = tl.sum(P, 1)
    l_i = l_i * correction + l_ij
    O_block = O_block * correction [:, None]
    m_i = m_ij

    #compute PV and add to accum (N, N) (N, C) -> (N, C)
    PV = tl.dot(P.to(v_block.dtype), v_block)
    O_block += PV


  #once done
  return m_i, l_i, O_block


@triton.autotune(
    configs=configs, key=["HEAD_DIM_Q", "SEQ_LEN", "NUM_HEAD"]
)
@triton.jit
def _forward_q(
      q,
      k,
      v,
      out,
      M,
      sqrt_dk,
      SEQ_LEN,
      HEAD_DIM_Q: tl.constexpr,
      NUM_HEAD,
      stage: tl.constexpr,
      BLOCK_SIZE_M: tl.constexpr,
      BLOCK_SIZE_N: tl.constexpr,
      num_warps: tl.constexpr,
      num_stages: tl.constexpr,
):
  tl.static_assert(BLOCK_SIZE_M % BLOCK_SIZE_N == 0)
  q_group = tl.program_id(0) #q position
  batch_head_index = tl.program_id(1) #where our batch head is

  #locate where q is
  batch_head_start = batch_head_index * SEQ_LEN * HEAD_DIM_Q
  q_start = q + batch_head_start + q_group * BLOCK_SIZE_M * HEAD_DIM_Q

  q_offsets = (
                tl.arange(0, BLOCK_SIZE_M)[:, None] * HEAD_DIM_Q +
                tl.arange(0, HEAD_DIM_Q)[None, :]
  )

  q = q_start + q_offsets


  #initialize our local m_i and l_i
  m_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) - float("inf")
  l_i = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32) + 1.0
  O_block = tl.zeros([BLOCK_SIZE_M, HEAD_DIM_Q], dtype=tl.float32)

  #generate masks load our Q block
  q_rows = q_group * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  mask = q_rows[:, None] < SEQ_LEN
  Q = tl.load(q, mask=mask, other=0.0)

  #divide by lg2 so we can use exp2 for later ops
  sqrt_dk *= 1.44269504  # 1/log(2)

  #first we need to process the kv blocks that occur before the corresponding q block. if its non-causal we can just process the whole kv
  m_i, l_i, O_block = _forward_inner_kv(
      Q,
      k,
      v,
      O_block,
      sqrt_dk,
      m_i,
      l_i,
      SEQ_LEN,
      HEAD_DIM_Q,
      stage,
      BLOCK_SIZE_M,
      BLOCK_SIZE_N,
      num_stages,
  )

  #then we need to process the kv blocks that lie in the same q block if its causal
  if stage == 3:
    m_i, l_i, O_block = _forward_inner_kv(
      Q,
      k,
      v,
      O_block,
      sqrt_dk,
      m_i,
      l_i,
      SEQ_LEN,
      HEAD_DIM_Q,
      2,
      BLOCK_SIZE_M,
      BLOCK_SIZE_N,
      num_stages,
    )

  #now we store our pointers
  m_i += tl.math.log2(l_i)
  O_block = O_block / l_i[:, None]
  
  M_ptr = (
            M + 
            batch_head_index * SEQ_LEN + 
            q_group * BLOCK_SIZE_M
          )
  
  M_offsets = tl.arange(0, BLOCK_SIZE_M)
  M_ptr = M_ptr + M_offsets

  o_ptr = (
            out + 
            batch_head_start + 
            q_group * BLOCK_SIZE_M * HEAD_DIM_Q + 
            q_offsets)

  tl.store(M_ptr, m_i, mask=q_rows<SEQ_LEN)
  tl.store(o_ptr, O_block, mask=mask)
