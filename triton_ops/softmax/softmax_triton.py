import torch
import torch.nn as nn
import triton
import triton.language as tl
from triton.runtime import driver

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


@triton.autotune(
    [
        triton.Config({}, num_warps=w)
        for w in [2, 4, 8, 16]
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
):
  """
  we compute the softmax for a row
  our grid is (n_row, 1, 1), so we just find the position of the row and compute
  """

  #process start of row
  row_index = tl.program_id(0)
  row_pointer = input_pointer + row_index * input_stride

  #capture the row and masks
  col_offsets = tl.arange(0, BLOCK_SIZE)
  row = row_pointer + col_offsets
  mask = col_offsets < n_cols
  row_values = tl.load(row, mask=mask, other=float("-inf")).to(tl.float32)

  #after row is loaded, we normalize the row before exponentiating it
  row_minus_max = row_values - tl.max(row_values, axis=0)
  numerator = tl.exp(row_minus_max)

  #we find the denominator and softmax
  denominator = tl.sum(numerator, axis=0)
  softmax_result = numerator/denominator

  output_row_pointer = output_pointer + row_index * output_stride
  output_row = output_row_pointer + col_offsets
  tl.store(output_row, softmax_result.to(input_pointer.dtype.element_ty), mask=mask)


@triton.autotune(
    [
        triton.Config({}, num_warps=w)
        for w in [2, 4, 8, 16]
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
):
  """
  we compute the partial derivative of S wrt to x_i, which influences every value in each softmax output
  we then find dLoss/dx_i given the derivative calculated and write this back
  """

  row_start = tl.program_id(0)
  row_step = tl.num_programs(0)

  y_row_pointer = y_pointer + row_start * y_row_stride
  col_offset = tl.arange(0, BLOCK_SIZE)
  mask = col_offset < n_cols
  y_index = y_row_pointer + col_offset

  #set our dy pointers and mask
  dy_row_pointer = dy_pointer + row_start * y_row_stride
  dy_index = dy_row_pointer + col_offset

  #load our y and dy
  dy = tl.load(dy_index, mask=mask, other=0.0).to(tl.float32)
  y = tl.load(y_index, mask=mask, other=0.0).to(tl.float32) #we set other to 0 as we are running tl.sum(), so we don't want the padded values to affect our division

  #calculate our dx
  dx = y * (dy - tl.sum(y * dy, axis=0))

  #calculate our dx pointer
  dx_arrays = dx_pointer + row_start * x_row_stride + col_offset
  tl.store(dx_arrays, dx.to(dx_pointer.dtype.element_ty), mask=mask)



class Softmax(torch.autograd.Function):

  @staticmethod
  def forward(ctx, x):
    """
    Launch a grid of size (row, 1, 1) so that each SM processes one softmax row in parallel
    num_stages and num_warps are autotuned
    """
    original_shape = x.shape
    x_reshaped = x.reshape(-1, x.shape[-1]) #convert into 2d 
    n_rows, n_cols = x_reshaped.shape

    #we create our output tensor and allocate BLOCK_SIZE
    y = torch.empty_like(x_reshaped)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    #we determine row strides
    x_stride = x_reshaped.stride(0)
    y_stride = y.stride(0)

    #autotune num_stage and num_warps
    softmax_kernel[(n_rows, 1, 1)](
        x_reshaped,
        y,
        x_stride,
        y_stride,
        n_rows,
        n_cols,
        BLOCK_SIZE,
    )

    y = y.reshape(original_shape)
    ctx.save_for_backward(y)
    ctx.original_shape = original_shape

    return y

  @staticmethod
  def backward(ctx, dy):
    """
    Launch a grid of size (row, 1, 1) so that each SM processes one dx row in parallel
    num_stages and num_warps are autotuned
    """
    #get back our y and shape
    original_shape = ctx.original_shape
    y, = ctx.saved_tensors

    #reshape our y, dy and x
    dy_reshaped = dy.reshape(-1, dy.shape[-1]) #convert to 2d
    y_reshaped = y.reshape(-1, dy.shape[-1])
    n_rows, n_cols = dy_reshaped.shape

    #create our output dx tensor and BLOCK SIZE
    dx = torch.empty_like(y_reshaped)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    #get stride for dy and dx
    y_row_stride = dy_reshaped.stride(0)
    x_row_stride = dx.stride(0)

    #num stages and num warps are autotuned
    softmax_bwd[(n_rows, 1, 1)](
        y_reshaped,
        dx,
        dy_reshaped,
        y_row_stride,
        x_row_stride,
        n_rows,
        n_cols,
        BLOCK_SIZE,
    )

    dx = dx.reshape(original_shape)

    return dx


def test_softmax():
  #initialize softmax layers. we don't initialize a softmax instance for triton. rather, we just call the class. this is because its a static method and .apply integrates with torch.autograd
  x = torch.randn(10, 10, device='cuda', requires_grad=True)

  # forward
  y_triton = Softmax.apply(x)
  y_torch = torch.softmax(x, dim=-1)
  print("Forward match:", torch.allclose(y_triton, y_torch, atol=1e-5))

  # backward
  dy = torch.randn_like(x)
  y_triton.backward(dy)
  grad_triton = x.grad.clone()

  x.grad = None
  y_torch = torch.softmax(x, dim=-1)
  y_torch.backward(dy)
  grad_torch = x.grad.clone()
  print("Backward match:", torch.allclose(grad_triton, grad_torch, atol=1e-5))
  print("Max diff:", (grad_triton - grad_torch).abs().max().item())

