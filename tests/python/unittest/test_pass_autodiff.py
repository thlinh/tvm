import tvm
import topi
import numpy as np
from nnvm.testing.check_computation import check_numerical_grads

def get_shape(tensor):
    return [s.value for s in tensor.shape]

# A helper checking the gradient of sum(out) wrt inp
def test_grad(out, inp, args):
    sout = tvm.create_schedule(out.op)
    mout = tvm.build(sout, [out, inp] + args)

    ones = tvm.compute(out.shape, lambda *inds: 1.0)

    jac = tvm.ir_pass.JacobianRecursive(out, inp, ones)

    print("Jacobian/grad body:")
    print(jac.op.body)
    print("")

    sjac = tvm.create_schedule(jac.op)
    mjac = tvm.build(sjac, [jac, inp] + args)

    def fun(**kwargs):
        aaa = [tvm.nd.empty(get_shape(out), out.dtype)] + \
            [tvm.ndarray.array(kwargs[a.name]) for a in [inp] + args]
        mout(*aaa)
        return aaa[0].asnumpy().sum()

    arg_vals = {a.name: np.random.uniform(-10, 10, size=get_shape(a)).astype(a.dtype)
                for a in [inp] + args}
    arg_vals_lst = [tvm.ndarray.array(arg_vals[a.name]) for a in [inp] + args]

    j_arg_vals = [tvm.nd.empty(get_shape(inp), jac.dtype)] + arg_vals_lst
    mjac(*j_arg_vals)
    j_res = j_arg_vals[0].asnumpy()

    check_numerical_grads(fun, arg_vals, {inp.name: j_res})

# Test some simple expressions
def test_autodiff():
    x = tvm.var("x", dtype='float32')
    k = tvm.reduce_axis((0, 10), name="k")
    l = tvm.reduce_axis((0, 10), name="l")
    A0 = tvm.placeholder((10, 10), name='A0')
    A1 = tvm.placeholder((10, 10), name='A1')

    B = tvm.compute((10, 10), lambda i, j: A0[i, j] + A0[j, i], name='B')
    test_grad(B, A0, [])

    B = tvm.compute((10, 10), lambda i, j: A0[i, j] * A0[j, i], name='B')
    test_grad(B, A0, [])

    B = tvm.compute((10, 10), lambda i, j: tvm.sum(A0[i, k]*A0[k, i] + 5, axis=k), name='B')
    test_grad(B, A0, [])

    B = tvm.compute((10, 10), lambda i, j: tvm.max(A0[i, k]*A0[k, j] + 5, axis=k), name='B')
    test_grad(B, A0, [])

    B = tvm.compute((10, 10), lambda i, j: A0[i, j] * (A1[j, i] + A0[j, i]), name='B')
    test_grad(B, A0, [A1])

    B = tvm.compute((10, 10), lambda i, j: tvm.sum(A0[k, k] - A0[j + k, j]*A0[i, k], axis=k), name='B')
    test_grad(B, A0, [])

    def fcombine(x, y):
        return x*y

    def fidentity(t0):
        return tvm.const(1, t0)

    prod = tvm.comm_reducer(fcombine, fidentity, name='prod')
    B = tvm.compute((10, 10), lambda i, j: prod(A0[i, k] + A0[k, i], axis=k), name='B')
    test_grad(B, A0, [])

def test_nn_autodiff():
    X = tvm.placeholder((1, 2, 4, 4), name='X')
    W = tvm.placeholder((5, 2, 3, 3), name='W')
    W1 = tvm.placeholder((2, 5, 3, 3), name='W1')

    R = topi.nn.conv2d(X, W, 1, 1)
    test_grad(R, X, [W])
    test_grad(R, W, [X])

    R1 = topi.nn.conv2d(topi.nn.relu(R), W1, 1, 0)
    test_grad(R1, X, [W, W1])
    test_grad(R1, W, [X, W1])
    test_grad(R1, W1, [X, W])

if __name__ == "__main__":
    test_autodiff()
    test_nn_autodiff()
