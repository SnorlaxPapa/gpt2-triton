import torch.nn as nn
import triton
import triton.language as tl


@triton.jit
def _backward_dq_inner(
    q_tile,
    max,
    delta,
    dq_tile,
    do_tile,
    k_idx,
    v_idx,
    q_row,
    start_row,
    num_steps,
    N,
    C: tl.constexpr,
    BLOCK_SIZE_M2: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
    num_stages: tl.constexpr,
    MASK: tl.constexpr,
):
  row_offs = tl.arange(0, BLOCK_SIZE_N2)
  col_offs = tl.arange(0, C)

  #we have q tile, max, delta, now need capture kt and vt tiles
  kT_idx = (
            k_idx + 
            start_row * C +
            row_offs[None, :] * C +
            col_offs[:, None]
           )
  vT_idx = (
            v_idx + 
            start_row * C +
            row_offs[None, :] * C +
            col_offs[:, None]
           )
  
  k_row = start_row + tl.arange(0, BLOCK_SIZE_N2)
 
  for i in tl.range(num_steps, num_stages=num_stages):
    #load our kt and vt block
    row_mask = k_row < N
    kT_tile = tl.load(kT_idx, mask=row_mask[None, :], other=0.0)
    vT_tile = tl.load(vT_idx, mask=row_mask[None, :], other=0.0)

    #compute dQ
    P = tl.math.exp2(tl.dot(q_tile, kT_tile) - max)
    P = tl.where(k_row[None, :] < N, P, 0.0)

    if MASK:
      mask = q_row[:, None] >= k_row[None, :]
      P = tl.where(mask, P, 0.0)

    dP = tl.dot(do_tile, vT_tile)
    dS = P * (dP - delta[:, None])
    dS = dS.to(tl.float16)

    #accumulate dq
    dq_tile += tl.dot(dS, tl.trans(kT_tile))

    kT_idx += BLOCK_SIZE_N2 * C
    vT_idx += BLOCK_SIZE_N2 * C
    k_row += BLOCK_SIZE_N2
    

  return dq_tile


@triton.autotune(
    configs=[
      triton.Config(
        {'BLOCK_SIZE_M2': BM,
         'BLOCK_SIZE_N2': BN},
        num_warps = w,
        num_stages = s,
      )
      for BM in [64, 128, 256]
      for w in [4, 8]
      for BN in [32, 64, 128, 256]
      for s in [4, 5, 6, 7]
      if BM % BN == 0
    ], key=["N", "C", "causal"])
@triton.jit
def _backward_dq_outer(
    q,
    k,
    v,
    N,
    dq,
    C: tl.constexpr,
    sqrt_dk,
    dO,
    M,
    D,
    causal: tl.constexpr,
    BLOCK_SIZE_M2: tl.constexpr,
    BLOCK_SIZE_N2: tl.constexpr,
    num_warps: tl.constexpr,
    num_stages: tl.constexpr,
):
  #capture pointer
  log2e_scale = 1.44269504
  M_GROUP = tl.program_id(0)
  batch_head_index = tl.program_id(1)
  batch_head_stride = N * C

  row_off = tl.arange(0, BLOCK_SIZE_M2)
  col_off = tl.arange(0, C)
  rows = M_GROUP * BLOCK_SIZE_M2 + row_off

  #calc dq now. q centric loop based on block_m and block_n2
  #capture our q block, delta, and max for this block
  q_idx = (
            q +
            batch_head_index * batch_head_stride +
            rows[:, None] * C +
            col_off[None, :]
          )
  m_idx = (
            M +
            batch_head_index * N +
            rows
          )
  d_idx = (
            D +
            batch_head_index * N +
            rows
          )
  dO_idx = (
            dO +
            batch_head_index * batch_head_stride +
            rows[:, None] * C +
            col_off[None, :]
           )
  

  row_mask = rows[:, None] < N

  #load our q, m, dO and delta
  q_tile = (tl.load(q_idx, mask=row_mask, other=0.0) * log2e_scale * sqrt_dk).to(tl.float16)
  dq_tile = tl.zeros_like(q_tile)
  dq_tile = dq_tile.to(tl.float32)

  max = tl.load(m_idx, mask=rows < N, other=0.0)[:, None]
  delta = tl.load(d_idx, mask=rows < N, other=0.0)
  do_tile = tl.load(dO_idx, mask=row_mask, other=0.0)

  #prepare relevant pointers
  k_idx = k + batch_head_index * batch_head_stride
  v_idx = v + batch_head_index * batch_head_stride

  start_row = M_GROUP * BLOCK_SIZE_M2
  num_steps = tl.cdiv(N, BLOCK_SIZE_N2)

  if causal:
    num_steps = tl.cdiv(BLOCK_SIZE_M2, BLOCK_SIZE_N2)
    dq_tile = _backward_dq_inner(
      q_tile=q_tile,
      max=max,
      delta=delta,
      dq_tile=dq_tile,
      do_tile=do_tile,
      k_idx=k_idx,
      v_idx=v_idx,
      q_row=rows,
      start_row=start_row,
      num_steps=num_steps,
      N=N,
      C=C,
      BLOCK_SIZE_M2=BLOCK_SIZE_M2,
      BLOCK_SIZE_N2=BLOCK_SIZE_N2,
      MASK=True,
      num_stages=num_stages
    )

    num_steps = tl.cdiv(M_GROUP * BLOCK_SIZE_M2, BLOCK_SIZE_N2)


  #outer non causal function call with num_Steps
  dq_tile = _backward_dq_inner(
      q_tile=q_tile,
      max=max,
      delta=delta,
      dq_tile=dq_tile,
      do_tile=do_tile,
      k_idx=k_idx,
      v_idx=v_idx,
      q_row=rows,
      start_row=0,
      num_steps=num_steps,
      N=N,
      C=C,
      BLOCK_SIZE_M2=BLOCK_SIZE_M2,
      BLOCK_SIZE_N2=BLOCK_SIZE_N2,
      MASK=False,
      num_stages=num_stages
    )

  dq_tile *= sqrt_dk
  #store our dq
  dq_idx = (
            dq +
            batch_head_index * batch_head_stride +
            rows[:, None] * C +
            col_off[None, :]
          )
  
  tl.store(dq_idx, dq_tile, mask=row_mask)


@triton.jit
def _backward_dk_dv_inner(
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
    num_steps,
    num_stages: tl.constexpr,
    start_q,
    start_k,
    BLOCK_SIZE_N1: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    MASK: tl.constexpr,
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

  #Calculate and accumulate dk and dv for this block
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

    #increment
    qT_idx +=  BLOCK_SIZE_N1 * C
    delta_idx += BLOCK_SIZE_N1
    m_idx += BLOCK_SIZE_N1
    dO_idx += BLOCK_SIZE_N1 * C

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
    ], key=["N", "C", "causal"])
@triton.jit
def _backward_dkdv_outer(
    q,
    k,
    v,
    N,
    dk,
    dv,
    dq,
    C: tl.constexpr,
    sqrt_dk,
    out,
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

  batch_head_stride = N * C
  row_off = tl.arange(0, BLOCK_SIZE_M)
  col_off = tl.arange(0, C)

  rows = M_GROUP * BLOCK_SIZE_M + row_off
  row_mask = rows[:, None] < N

  M += batch_head_index * N
  D += batch_head_index * N
  dO += batch_head_index * N * C
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
  dk_tile = tl.zeros_like(k_tile)
  dv_tile = tl.zeros_like(v_tile)
  dk_tile = dk_tile.to(tl.float32)
  dv_tile = dv_tile.to(tl.float32)

  #process dv dk first
  q_start = 0
  k_start = M_GROUP * BLOCK_SIZE_M
  if causal:
    num_steps = BLOCK_SIZE_M // BLOCK_SIZE_N1
    #causal, meaning we only process q tiles that >= kv in the sequence
    q_start = k_start

    dk_tile, dv_tile = _backward_dk_dv_inner(
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
      num_steps=num_steps,
      num_stages=num_stages,
      start_q=q_start,
      start_k=k_start,
      BLOCK_SIZE_N1=BLOCK_SIZE_N1,
      BLOCK_SIZE_M=BLOCK_SIZE_M,
      MASK=True,
    )

    #increment our q start ot after k_start
    q_start += (num_steps * BLOCK_SIZE_N1)


  #fn call for dk, dv, this time masking off, q start at appropriate place
  remaining = tl.maximum(N - q_start, 0)
  num_steps = tl.cdiv(remaining, BLOCK_SIZE_N1)
  dk_tile, dv_tile = _backward_dk_dv_inner(
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
      num_steps=num_steps,
      num_stages=num_stages,
      start_q=q_start,
      start_k=k_start,
      BLOCK_SIZE_N1=BLOCK_SIZE_N1,
      BLOCK_SIZE_M=BLOCK_SIZE_M,
      MASK=False,
  )
  dk_tile *= sqrt_dk

  #store our dk_tile and dv_tile
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