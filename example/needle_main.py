import sys
sys.path.append("/workspace/needle/python")

import needle as ndl

x1 = ndl.Tensor([3], dtype="float32")
x2 = ndl.Tensor([4], dtype="float32")
x3 = x1 * x2
print(x3)

print(x3.op.gradient_as_tuple(ndl.Tensor([0.5], dtype="float32"),
    x3,
))
print(x3.op)
print(x3.op.gradient)
print(x3.op.gradient.__code__)

x2 = x1 ** 2
print(x2)
print(x2.op.gradient_as_tuple(ndl.Tensor([0.5], dtype="float32"),
    x2,
))