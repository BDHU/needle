"""Operator and gradient implementations."""
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp

import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api

r"""
get the identity NDArray for any given array
"""
def _identity(x: NDArray):
    if x is None:
        return x
    return x

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return self.scalar * a

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        return (out_grad * self.scalar * (input ** (self.scalar - 1)))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * (-lhs / (rhs) ** 2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # return array_api.divide(out_grad, self.scalar)
        return out_grad / self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)

class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        if self.axes is not None and len(self.axes) > 2:
            raise NotImplementedError

        if self.axes is None:
            # default to the last two dim
            if a.ndim >= 2:
                return array_api.swapaxes(a, a.ndim - 2, a.ndim - 1)
            else:
                return a

        if len(self.axes) > a.ndim or len(self.axes) < 2:
            return a

        return array_api.swapaxes(a, self.axes[0], self.axes[1])
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        if self.axes is None:
            if out_grad.ndim >= 2:
                return transpose(out_grad, out_grad.ndim - 2, out_grad.ndim - 1)
            else:
                return out_grad
        
        if len(self.axes) > out_grad.ndim or len(self.axes) < 2:
            return out_grad
            
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        return reshape(out_grad, input.shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        diff = out_grad.ndim - input.ndim
        # Force input and out_grad to take the same number of shape dim
        input_padded_shape = tuple(array_api.ones((diff,), dtype=int)) + input.shape
        sum_axes = tuple([i for i, (g, s) in enumerate(zip(input_padded_shape, out_grad.shape)) if g!=s])
        return reshape(summation(out_grad, sum_axes), input.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        i_s = array_api.array(input.shape)

        # reshape out_grad to have the same ndim as input
        if self.axes is not None:
            i_s[array_api.array(self.axes)] = int(1)
            out_grad = reshape(out_grad, tuple(i_s))

        return broadcast_to(out_grad, input.shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        batch_size = out_grad.shape[0]
        A, B = node.inputs

        if batch_size == A.shape[0] and batch_size == B.shape[0]:
            return matmul(out_grad, transpose(B)), matmul(transpose(A), out_grad)
        elif batch_size == A.shape[0] and batch_size != B.shape[0]:
            B_grad = matmul(transpose(A), out_grad)
            sum_id = B_grad.shape.index(B.shape[0])
            while sum_id > 0:
                sum_id -= 1
                B_grad = summation(B_grad, 0)
            return matmul(out_grad, transpose(B)), B_grad
        elif batch_size != A.shape[0] and batch_size == B.shape[0]:
            A_grad = matmul(out_grad, transpose(B))
            sum_id = A_grad.shape.index(A.shape[0])
            while sum_id > 0:
                sum_id -= 1
                A_grad = summation(A_grad, 0)
            return A_grad, matmul(transpose(A), out_grad)
        else:
            raise RuntimeError()
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        return out_grad / input
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        return out_grad * exp(input)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(0, a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        input, = node.inputs
        assert(input.shape == out_grad.shape)
        mask = zeros(input.shape)
        mask = [1 for (i, j) in zip(mask, input) if j >= 0]
        return multiply(mask, out_grad)
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)

def softmax(a):
    exps = exp(a)
    return exps / summation(exps, axes=0)

def logsoftmax(a):
    return log(softmax(a))

# class LogSoftmax(TensorOp):
#     def compute(self, Z):
#         ### BEGIN YOUR SOLUTION
#         exps = array_api.exp(Z - Z.max())
#         return array_api.log(exps / array_api.sum(exps))
#         ### END YOUR SOLUTION

#     def gradient(self, out_grad, node):
#         ### BEGIN YOUR SOLUTION
#         input, = node.inputs
#         return out_grad * logsoftmax(input) * (1.0 - (exp(input) / summation(input, axes=0)))
#         ### END YOUR SOLUTION


# def logsoftmax(a):
#     return LogSoftmax()(a)


# additional helper functions
def full(
    shape, fill_value, *, rand={}, dtype="float32", device=None, requires_grad=False
):
    # numpy do not need device argument
    kwargs = {"device": device} if array_api is not numpy else {}
    device = device if device else cpu()

    if not rand or "dist" not in rand:
        arr = array_api.full(shape, fill_value, dtype=dtype, **kwargs)
    else:
        if rand["dist"] == "normal":
            arr = array_api.randn(
                shape, dtype, mean=rand["mean"], std=rand["std"], **kwargs
            )
        if rand["dist"] == "binomial":
            arr = array_api.randb(
                shape, dtype, ntrials=rand["trials"], p=rand["prob"], **kwargs
            )
        if rand["dist"] == "uniform":
            arr = array_api.randu(
                shape, dtype, low=rand["low"], high=rand["high"], **kwargs
            )

    return Tensor.make_const(arr, requires_grad=requires_grad)


def zeros(shape, *, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, dtype=dtype, device=device, requires_grad=requires_grad)


def randn(
    shape, *, mean=0.0, std=1.0, dtype="float32", device=None, requires_grad=False
):
    return full(
        shape,
        0,
        rand={"dist": "normal", "mean": mean, "std": std},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randb(shape, *, n=1, p=0.5, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "binomial", "trials": n, "prob": p},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randu(shape, *, low=0, high=1, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "uniform", "low": low, "high": high},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 0, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 1, dtype=array.dtype, device=device, requires_grad=requires_grad
    )
