import tvm
import numpy as np
from nnvm.testing.check_computation import check_numerical_grads

def test_grad(out, inp, args, shapes):
    sout = tvm.create_schedule(out.op)
    mout = tvm.build(sout, [out, inp] + args)

    jac = tvm.ir_pass.Jacobian(out, inp)
    sjac = tvm.create_schedule(jac.op)
    mjac = tvm.build(sjac, [jac, inp] + args)

    print(jac.op.body)
    print("")

    def fun(**kwargs):
        aaa = [tvm.nd.empty(shapes[out.name], out.dtype)] + \
            [tvm.ndarray.array(kwargs[a.name]) for a in [inp] + args]
        mout(*aaa)
        return aaa[0].asnumpy().sum()

    arg_vals = {a.name: np.random.uniform(-10, 10, size=shapes[a.name]).astype(a.dtype)
                for a in [inp] + args}
    arg_vals_lst = [tvm.ndarray.array(arg_vals[a.name]) for a in [inp] + args]

    j_arg_vals = [tvm.nd.empty(shapes[out.name] + shapes[inp.name], jac.dtype)] + arg_vals_lst
    mjac(*j_arg_vals)
    j_res = j_arg_vals[0].asnumpy().sum(axis=tuple(range(len(shapes[out.name]))))

    check_numerical_grads(fun, arg_vals, {inp.name: j_res})

def test_autodiff():
    n = tvm.var("n")
    m = tvm.var("m")
    x = tvm.var("x", dtype='float32')
    k = tvm.reduce_axis((0, tvm.min(n, m)), name="k")
    l = tvm.reduce_axis((0, tvm.min(n, m)), name="l")
    A0 = tvm.placeholder((m, n), name='A0')
    A1 = tvm.placeholder((m, n), name='A1')

    B = tvm.compute((m, n), lambda i, j: A0[i, j] + A0[j, i], name='B')
    test_grad(B, A0, [], {'A0': (10,10), 'B': (10,10)})

    B = tvm.compute((m, n), lambda i, j: A0[i, j] * A0[j, i], name='B')
    test_grad(B, A0, [], {'A0': (10,10), 'B': (10,10)})

    B = tvm.compute((m, n), lambda i, j: tvm.sum(A0[i, k]*A0[k, i] + 5, axis=k), name='B')
    test_grad(B, A0, [], {'A0': (10,10), 'B': (10,10)})

    B = tvm.compute((m, n), lambda i, j: tvm.max(A0[i, k]*A0[k, j] + 5, axis=k), name='B')
    test_grad(B, A0, [], {'A0': (10,10), 'B': (10,10)})

    B = tvm.compute((m, n), lambda i, j: A0[i, j] * (A1[j, i] + A0[j, i]), name='B')
    test_grad(B, A0, [A1], {'A0': (10,10), 'A1': (10,10), 'B': (10,10)})

    B = tvm.compute((m, n), lambda i, j: tvm.sum(A0[k, k] - A0[j + k, j]*A0[i, k], axis=k), name='B')
    test_grad(B, A0, [], {'A0': (10,2), 'B': (10,2)})

    def fcombine(x, y):
        return x*y

    def fidentity(t0):
        return tvm.const(1, t0)

    prod = tvm.comm_reducer(fcombine, fidentity, name='prod')
    B = tvm.compute((m, n), lambda i, j: prod(A0[i, k] + A0[k, i], axis=k), name='B')
    test_grad(B, A0, [], {'A0': (10,10), 'B': (10,10)})

if __name__ == "__main__":
    test_autodiff()
