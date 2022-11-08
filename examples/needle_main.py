import sys
sys.path.append("/workspace/python")

import needle as ndl

import numpy as np

def numerical_grads(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    return numerical_grads

def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    print("the hell??????????????????????")
    if not backward:
        out = f(*args, **kwargs)
        print(out.op.gradient_as_tuple(ndl.Tensor(np.ones(out.shape)), out))
        print("end")
        computed_grads = [x.numpy() for x in out.op.gradient_as_tuple(ndl.Tensor(np.ones(out.shape)), out)]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]

    print("numerical grad:")
    print(numerical_grads)

    print("computed_grads:")
    print(computed_grads)
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i])
        for i in range(len(args))
    )
    assert error < tol
    return computed_grads

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

# power scalar
print("================PowerScalar================")
x3 = x1 ** 2
print(x3.op.gradient_as_tuple(ndl.Tensor([0.5], dtype="float32"), x3))

# devide element wise
print("================EleWiseDiv================")
x4 = x1 / x2
print(x4)
print(x4.op.gradient_as_tuple(ndl.Tensor([0.5], dtype="float32"), x4))

## tranpose
x1 = ndl.Tensor([[3, 2], [1, 2]], dtype="float32")
x2 = ndl.transpose(x1)
print("================Transpose================")
print("input:")
print(x1)
print("output:")
print(x2)
print("gradient:")
print(x2.op.gradient(ndl.Tensor([0.5], dtype="float32"), x2))

x1 = ndl.transpose(ndl.Tensor(np.random.randn(3, 5, 4)), (1, 2))
numerical = numerical_grads(ndl.transpose, ndl.Tensor(np.random.randn(3, 5, 4)), axes=(1, 2))
print(numerical)
print(len(numerical))
print(numerical[0].shape)
# gradient_check(ndl.transpose, ndl.Tensor(np.random.randn(3, 5, 4)), axes=(1, 2))
print("-------------------")
print(x1.op.gradient(ndl.Tensor([0.5], dtype="float32"), x1))

gradient_check(ndl.transpose, ndl.Tensor(np.random.randn(2, 5, 4)), axes=(1, 2))

# gradient_check(ndl.divide_scalar, ndl.Tensor(np.random.randn(5, 4)), scalar=0.5)