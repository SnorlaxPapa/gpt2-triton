import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.runtime import driver

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


@triton.autotune(
    [
        triton.Config({"num_stages": s}, num_warps=w)
        for s in [1, 2, 3, 4, 5, 6] for w in [2, 4, 8, 16]
    ],
    key=["n_cols", "n_rows", "BLOCK_SIZE"]
)
@triton.jit
def softmax_kernel(
  input_pointer,
  output_pointer,
  input_stride,
  output_stride,
  n_rows,
  n_cols,
  BLOCK_SIZE: tl.constexpr,
  num_stages: tl.constexpr,  
):
  """
  we compute the softmax for a row
  we iterate in the form for each program i, i + num_programs, i + 2 * num_programs and write the output back to HBM
  """

  row_start = tl.program_id(0)
  row_step = tl.num_programs(0)

  #we iterate in the form 1, 11, 21, 31 etc for better indexing
  for row_index in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
    #we extract where the row is first 
    row_pointer = input_pointer + row_index * input_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    row = row_pointer + col_offsets

    #we initialize a mask where col < n_col to filter out extra columns
    mask = col_offsets < n_cols
    row_values = tl.load(row, mask=mask, other=float("-inf")).to(tl.float32) #we set false values to -inf so these dont affect our calculations
    #after row is loaded, we find normalize the row before exponentiating it
    row_minus_max = row_values - tl.max(row_values, axis=0)
    numerator = tl.exp(row_minus_max)

    #we finally softmax the row
    denominator = tl.sum(numerator, axis=0)
    softmax_result = numerator/denominator

    #then we write back to our output row
    output_row_pointer = output_pointer + row_index * output_stride
    output_row = output_row_pointer + col_offsets
    tl.store(output_row, softmax_result.to(input_pointer.dtype.element_ty), mask=mask)



@triton.autotune(
    [
        triton.Config({"num_stages": s}, num_warps=w)
        for s in [1, 2, 3, 4, 5, 6] for w in [2, 4, 8, 16]
    ],
    key=["n_cols", "n_rows", "BLOCK_SIZE"]
)
@triton.jit
def softmax_bwd(
    y_pointer,
    dx_pointer,
    dy_pointer,
    y_row_stride,
    x_row_stride,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
  """
  we compute the partial derivative of S wrt to x_i, which influences every value in each softmax output
  we then find dLoss/dx_i given the derivative calculated and write this back 
  """

  row_start = tl.program_id(0)
  row_step = tl.num_programs(0)

  #iterate through rows like in forward
  for row_index in tl.range(row_start, n_rows, row_step, num_stages=num_stages):
    #set our y pointers and mask
    y_row_pointer = y_pointer + row_index * y_row_stride
    col_offset = tl.arange(0, BLOCK_SIZE)
    mask = col_offset < n_cols
    y_index = y_row_pointer + col_offset

    #set our dy pointers and mask
    dy_row_pointer = dy_pointer + row_index * y_row_stride
    dy_index = dy_row_pointer + col_offset

    #load our y and dy
    dy = tl.load(dy_index, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_index, mask=mask, other=0.0).to(tl.float32) #we set other to 0 as we are running tl.sum(), so we don't want the padded values to affect our division

    #calculate our dx
    dx = y * (dy - tl.sum(y * dy, axis=0))

    #calculate our dx pointer
    dx_arrays = dx_pointer + row_index * x_row_stride + col_offset
    tl.store(dx_arrays, dx.to(dx_pointer.dtype.element_ty), mask=mask)



class Softmax(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x):
    """
    Launch a grid of size (row, 1, 1) so that each SM processes one softmax row in parallel
    num_stages and num_warps are autotuned
    """
    n_rows, n_cols = x.shape

    #we create our output tensor and allocate BLOCK_SIZE
    y = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    #we determine row strides
    x_stride = x.stride(0)
    y_stride = y.stride(0)

    #autotune num_stage and num_warps
    softmax_kernel[(n_rows, 1, 1)](
        x,
        y,
        x_stride,
        y_stride,
        n_rows,
        n_cols,
        BLOCK_SIZE,
    ) 
    ctx.save_for_backward(x, y)

    return y
 
  @staticmethod
  def backward(ctx, dy):
    """
    Launch a grid of size (row, 1, 1) so that each SM processes one dx row in parallel
    num_stages and num_warps are autotuned
    """
    #get back our y
    x, y = ctx.saved_tensors
    n_rows, n_cols = dy.shape

    #create our output dx tensor and BLOCK SIZE
    dx = torch.empty_like(x)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    #get stride for dy and dx
    y_row_stride = dy.stride(0)
    x_row_stride = dx.stride(0)

    #num stages and num warps are autotuned
    softmax_bwd[(n_rows, 1, 1)](
        y,
        dx,
        dy,
        y_row_stride,
        x_row_stride,
        n_rows,
        n_cols,
        BLOCK_SIZE,
    )

    return dx