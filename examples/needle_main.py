import sys
sys.path.append("/workspace/Desktop/needle/python")

import needle as ndl

import numpy as np

# print(np.power(ndl.Tensor([1, 2]), 2))

x1 = ndl.Tensor([3, 2], dtype="float32")
x2 = ndl.Tensor([1, 2], dtype="float32")
# x2 = ndl.Tensor([4], dtype="float32")
# x3 = x1 * x2
# print(x3)

# print(x3.op.gradient_as_tuple(ndl.Tensor([0.5], dtype="float32"),
#     x3,
# ))
# print(x3.op)
# print(x3.op.gradient)
# print(x3.op.gradient.__code__)

x3 = x1 ** 2

print(x3.op.gradient_as_tuple(ndl.Tensor([0.5], dtype="float32"), x3))

x4 = x1 / x2
print(x4)
print(x4.op.gradient_as_tuple(ndl.Tensor([0.5], dtype="float32"), x4))