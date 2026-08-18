import torch.nn as nn
import triton
import triton.language as tl

@triton.jit
def _backward_inner(
    q_idx,
    k_tile,
    v_tile,
    dO,
    M,
    D,
    N,
    C: tl.constexpr,
    dk_tile,
    dv_tile,
    dq,
    num_steps,
    num_stages: tl.constexpr,
    start_q,
    start_k,
    BLOCK_SIZE_N1: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    MASK: tl.constexpr,
    LN2: tl.constexpr,
):
  #locate the row our q_tile starts
  q_row = tl.arange(0, BLOCK_SIZE_N1)
  q_col = tl.arange(0, C)
  qT_idx = (
            q_idx +
            start_q * C +
            q_col[:, None] +
            q_row[None, :] * C
          )

  m_idx = start_q + q_row
  delta_idx = D + start_q + q_row
  k_rows = start_k + tl.arange(0, BLOCK_SIZE_M)
  dO_idx = (
              dO +
              start_q * C +
              q_col[None, :] +
              q_row[:, None] * C
           )

  dq_idx = (
              dq + 
              start_q * C +
              q_col[None, :] +
              q_row[:, None] * C
           )
  
  #Calculate and accumulate dk, dq and dv for this block
  for i in tl.range(num_steps, num_stages=num_stages):
    #load qT, and m
    row = start_q + i * BLOCK_SIZE_N1 + q_row
    qT = tl.load(qT_idx, mask=row[None, :] < N, other=0.0)
    m = tl.load(M + m_idx, mask=row < N, other=0.0)

    #calculations
    sT = tl.dot(k_tile, qT)#(QKT)T  = KQT
    pT = tl.math.exp2(sT - m[None, :])

    #mask positions where query > columns
    if MASK:
      mask = row[None, :] >= k_rows[:, None]
      pT = tl.where(mask, pT, 0.0)

    #load dO
    dO_tile = tl.load(dO_idx, mask=row[:, None] < N, other=0.0)

    #calculate dV
    ppT = pT
    ppT = ppT.to(tl.float16) #downcast to fp16 for faster matmul
    dv_tile += tl.dot(ppT, dO_tile)

    #load delta
    delta = tl.load(delta_idx, mask=row < N, other=0.0)

    #calc dpt, dst, dk
    dpT = tl.dot(v_tile, tl.trans(dO_tile))
    dsT = pT * (dpT - delta[None, :])
    dsT = dsT.to(tl.float16)
    dk_tile += tl.dot(dsT, tl.trans(qT)) 

    dq_tile = tl.dot(tl.trans(dsT), k_tile) * LN2
    tl.atomic_add(
      dq_idx,
      dq_tile,
      mask=row[:, None] < N,
      sem="relaxed",
    )

    #increment
    qT_idx +=  BLOCK_SIZE_N1 * C
    delta_idx += BLOCK_SIZE_N1
    m_idx += BLOCK_SIZE_N1
    dO_idx += BLOCK_SIZE_N1 * C
    dq_idx += BLOCK_SIZE_N1 * C

  return dk_tile, dv_tile


@triton.autotune(
    configs=[
      triton.Config(
        {'BLOCK_SIZE_M': BM,
         'BLOCK_SIZE_N1': BN},
        num_warps = w,
        num_stages = s,
      )
      for BM in [64, 128, 256]
      for w in [4, 8]
      for BN in [32, 64, 128, 256]
      for s in [4, 5, 6, 7]
      if BM % BN == 0
    ], key=["N", "C", "causal"], reset_to_zero=["dq"])
@triton.jit
def _backward_outer(
    q,
    k,
    v,
    N,
    dk,
    dv,
    dq,
    C: tl.constexpr,
    sqrt_dk,
    dO,
    M,
    D,
    causal: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N1: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
  #ok so we first snag our primary kv block and move our pointers
  M_GROUP = tl.program_id(0)
  batch_head_index = tl.program_id(1)
  LN2 = 0.6931471805599453

  batch_head_stride = N * C
  row_off = tl.arange(0, BLOCK_SIZE_M)
  col_off = tl.arange(0, C)

  rows = M_GROUP * BLOCK_SIZE_M + row_off
  row_mask = rows[:, None] < N

  M += batch_head_index * N
  D += batch_head_index * N
  dO += batch_head_index * N * C
  dq += batch_head_index * N * C
  #first we find our kv blocks
  k_idx = (
            k +
            batch_head_index * batch_head_stride +
            rows[:, None] * C +
            col_off[None, :]
          )

  v_idx = (
            v +
            batch_head_index * batch_head_stride +
            rows[:, None] * C +
            col_off[None, :]
          )

  q_idx = (
            q +
            batch_head_index * batch_head_stride
          )

  log2e_scale = 1.44269504
  #load kv
  k_tile = (tl.load(k_idx, mask=row_mask, other=0.0) * sqrt_dk * log2e_scale).to(tl.float16)
  v_tile = tl.load(v_idx, mask=row_mask, other=0.0)

  #set up our dk dv blocks
  dk_tile = tl.zeros_like(k_tile).to(tl.float32)
  dv_tile = tl.zeros_like(v_tile).to(tl.float32)

  #process dv dk first
  q_start = 0
  k_start = M_GROUP * BLOCK_SIZE_M
  if causal:
    num_steps = BLOCK_SIZE_M // BLOCK_SIZE_N1
    #causal, meaning we only process q tiles that >= kv in the sequence
    q_start = k_start

    dk_tile, dv_tile= _backward_inner(
      q_idx=q_idx,
      k_tile=k_tile,
      v_tile=v_tile,
      dO=dO,
      M=M,
      D=D,
      N=N,
      C=C,
      dk_tile=dk_tile,
      dv_tile=dv_tile,
      dq=dq,
      num_steps=num_steps,
      num_stages=num_stages,
      start_q=q_start,
      start_k=k_start,
      BLOCK_SIZE_N1=BLOCK_SIZE_N1,
      BLOCK_SIZE_M=BLOCK_SIZE_M,
      MASK=True,
      LN2 = LN2,
    )

    #increment our q start ot after k_start
    q_start += (num_steps * BLOCK_SIZE_N1)


  #fn call for dk, dv, this time masking off, q start at appropriate place
  remaining = tl.maximum(N - q_start, 0)
  num_steps = tl.cdiv(remaining, BLOCK_SIZE_N1)
  dk_tile, dv_tile = _backward_inner(
      q_idx=q_idx,
      k_tile=k_tile,
      v_tile=v_tile,
      dO=dO,
      M=M,
      D=D,
      N=N,
      C=C,
      dk_tile=dk_tile,
      dv_tile=dv_tile,
      dq=dq,
      num_steps=num_steps,
      num_stages=num_stages,
      start_q=q_start,
      start_k=k_start,
      BLOCK_SIZE_N1=BLOCK_SIZE_N1,
      BLOCK_SIZE_M=BLOCK_SIZE_M,
      MASK=False,
      LN2=LN2,
  )
  dk_tile *= sqrt_dk

  #store our dq_tile, dk_tile and dv_tile
  dk_idx = (
            dk +
            batch_head_index * batch_head_stride +
            rows[:, None] * C +
            col_off[None, :]
          )

  dv_idx = (
            dv +
            batch_head_index * batch_head_stride +
            rows[:, None] * C +
            col_off[None, :]
          )

  tl.store(dk_idx, dk_tile, mask=row_mask)
  tl.store(dv_idx, dv_tile, mask=row_mask)



@triton.autotune(
    configs=[
      triton.Config(
        {'BLOCK_SIZE_M': BM}, 
        num_warps = w,
      )
      for BM in [32, 64, 128, 256] for w in [2, 4, 8]
    ], key=["N", "C"])
@triton.jit
def _preprocess(
    out,
    dO,
    DD,  
    N,
    C: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
):
  #find start of current block grid (BLOCK_SIZE_M, BATCH, HEAD) -> each head got own N, N split into BLOCK_SIZE_M
  group_index = tl.program_id(0)
  batch_head_index = tl.program_id(1)

  batch_head_stride = N * C
  start_O = (
              out + batch_head_index * batch_head_stride + 
              group_index * BLOCK_SIZE_M * C
            )  
  start_dO = (
              dO + batch_head_index * batch_head_stride + 
              group_index * BLOCK_SIZE_M * C
             )

  #capture our block offsets
  O_offsets = tl.arange(0, BLOCK_SIZE_M)[:, None] * C + tl.arange(0, C)[None, :]

  rows = group_index * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
  mask = rows[:, None] < N

  #load
  dOut = tl.load(start_dO + O_offsets, mask=mask, other=0.0)
  O = tl.load(start_O + O_offsets, mask=mask, other=0.0)

  dd = tl.sum(dOut * O, axis=1)

  #write back to DD
  dd_idx = DD + batch_head_index * N + rows
  tl.store(dd_idx, dd, mask=rows < N)