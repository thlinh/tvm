# This example demonstrates Automatic Differentiation for TVM basic operations and TOPI primitives.
# See `test_autodiff()` and `test_nn_autodiff()` for details.

import tvm
import topi
import numpy as np
from nnvm.testing.check_computation import check_numerical_grads
import time

def get_shape(tensor):
    return [tvm.ir_pass.Simplify(s).value for s in tensor.shape]

# A helper checking the gradient of sum(out) wrt inp
def test_grad(out, inp, args=[], in_range=(-10,10)):
    if not isinstance(inp, (list, tuple)):
        inp = [inp]

    sout = tvm.create_schedule(out.op)
    mout = tvm.build(sout, [out] + inp + args)

    ones = topi.full_like(out, 1.0)

    t = time.time()
    jacs = list(tvm.ir_pass.JacobianRecursive(out, inp, ones))
    print("JAC TIME: ", time.time() - t)

    t = time.time()
    sjac = tvm.create_schedule([j.op for j in jacs])
    mjac = tvm.build(sjac, jacs + inp + args)
    print("BUILD TIME: ", time.time() - t)

    def fun(*arguments):
        aaa = [tvm.nd.empty(get_shape(out), out.dtype)] + [tvm.nd.array(a) for a in arguments]
        mout(*aaa)
        return aaa[0].asnumpy().sum()

    arg_vals = [tvm.nd.array(np.random.uniform(in_range[0], in_range[1],
                                               size=get_shape(a)).astype(a.dtype))
                for a in inp + args]

    j_arg_vals = [tvm.nd.empty(get_shape(i), j.dtype) for i, j in zip(inp, jacs)] + arg_vals
    t = time.time()
    mjac(*j_arg_vals)
    j_res = [j_arg_vals[j].asnumpy() for j, _ in enumerate(jacs)]
    print("JAC EXEC TIME: ", time.time() - t)

    t = time.time()
    check_numerical_grads(fun, [a.asnumpy() for a in arg_vals], j_res)
    print("NUMGRAD TIME: ", time.time() - t)

# Test some simple expressions
def test_autodiff():
    x = tvm.var("x", dtype='float32')
    k = tvm.reduce_axis((0, 10), name="k")
    l = tvm.reduce_axis((0, 10), name="l")
    A0 = tvm.placeholder((10, 10), name='A0')
    A1 = tvm.placeholder((10, 10), name='A1')

    B = tvm.compute((10, 10), lambda i, j: A0[i, j] + A0[j, i], name='B')
    test_grad(B, A0)

    B = tvm.compute((10, 10), lambda i, j: A0[i, j] + tvm.exp(A0[j, i]), name='B')
    test_grad(B, A0)

    B = tvm.compute((10, 10), lambda i, j: tvm.log(tvm.abs(A0[i, j] + tvm.exp(A0[j, i]))), name='B')
    test_grad(B, A0)

    B = tvm.compute((10, 10), lambda i, j: tvm.sigmoid(A0[i, j]*A0[i, j]*A0[j, i]), name='B')
    test_grad(B, A0)

    B = tvm.compute((10, 10), lambda i, j: tvm.tanh(A0[i, j]*A0[i, j]*A0[j, i]), name='B')
    test_grad(B, A0)

    B = tvm.compute((10, 10), lambda i, j: A0[i, j] * A0[j, i], name='B')
    test_grad(B, A0)

    B = tvm.compute((10, 10), lambda i, j: tvm.sum(A0[i, k]*A0[k, i] + 5, axis=k), name='B')
    test_grad(B, A0)

    B = tvm.compute((10, 10), lambda i, j: tvm.max(A0[i, k]*A0[k, j] + 5, axis=k), name='B')
    test_grad(B, A0)

    B = tvm.compute((10, 10), lambda i, j: A0[i, j] * (A1[j, i] + A0[j, i]), name='B')
    test_grad(B, A0, [A1])

    B = tvm.compute((10, 10), lambda i, j: tvm.sum(A0[k, k] - A0[tvm.min(j + k, 9), j]*A0[i, k],
                                                   axis=k),
                    name='B')
    test_grad(B, A0)

    def fcombine(x, y):
        return x*y

    def fidentity(t0):
        return tvm.const(1, t0)

    prod = tvm.comm_reducer(fcombine, fidentity, name='prod')
    B = tvm.compute((10, 10), lambda i, j: prod(A0[i, k] + A0[k, i], axis=k), name='B')
    test_grad(B, A0)

def test_topi_autodiff():
    X = tvm.placeholder((1, 2, 4, 4), name='X')
    W = tvm.placeholder((5, 2, 3, 3), name='W')
    W1 = tvm.placeholder((2, 5, 3, 3), name='W1')

    R = topi.nn.conv2d(X, W, 1, 1)
    test_grad(R, [X, W])

    R1 = topi.nn.conv2d(topi.nn.relu(R), W1, 1, 0)
    test_grad(R1, [X, W, W1])

    X = tvm.placeholder((1, 2, 5, 5), name='X')
    W = tvm.placeholder((2, 2, 3, 3), name='W')

    R = X + topi.nn.conv2d(X + topi.nn.conv2d(X, W, 1, 1), W, 1, 1)
    test_grad(R, [X, W])

    S = topi.nn.softmax(topi.reshape(R, (1, 32)))
    test_grad(S, [X, W])

    S = topi.sigmoid(topi.reshape(R, (1, 32)))
    test_grad(S, [X, W])

    S = topi.tanh(topi.reshape(R, (1, 32)))
    test_grad(S, [X, W])

    S = topi.nn.log_softmax(topi.reshape(R, (1, 32)))
    test_grad(S, [X, W])
    test_grad(S, [W], [X])

def test_some_conv2d_net():
    batch_size = 1
    num_classes = 10

    features = 4
    dense_units = 16

    x = tvm.placeholder((batch_size, 28, 14, 1))
    y = tvm.placeholder((batch_size, num_classes))

    w1 = tvm.placeholder((features, 1, 3, 5))
    b1 = tvm.placeholder((features,))
    w2 = tvm.placeholder((features, features, 3, 5))
    b2 = tvm.placeholder((features,))
    b3 = tvm.placeholder((dense_units,))
    w4 = tvm.placeholder((num_classes, dense_units))
    b4 = tvm.placeholder((num_classes,))

    t = topi.transpose(x, [0, 3, 1, 2])
    t = topi.nn.relu(topi.nn.conv2d(t, w1, 1, 0) + topi.reshape(b1, (1, features, 1, 1)))
    t = topi.nn.relu(topi.nn.conv2d(t, w2, 1, 0) + topi.reshape(b2, (1, features, 1, 1)))
    t = topi.nn.pool(t, [2, 2], [2, 2], [0, 0, 0, 0], 'avg')
    t = topi.transpose(t, [0, 2, 3, 1])
    t = topi.nn.flatten(t)
    w3 = tvm.placeholder((dense_units, get_shape(t)[1]))
    t = topi.nn.relu(topi.nn.dense(t, w3, b3))
    t = topi.nn.dense(t, w4, b4)

    t = - topi.sum(y * topi.nn.log_softmax(t)) / batch_size

    weights = [w1, b1, w2, b2, w3, b3, w4, b4]

    test_grad(t, weights, [x, y], in_range=(-1.0, 1.0))

# Just compare the general shape, ignoring names and functions
def _loosely_eq(stmt1, stmt2):
    if stmt1 == stmt2:
        return True
    elif isinstance(stmt1, tvm.stmt.AttrStmt):
        return _loosely_eq(stmt1.body, stmt2)
    elif isinstance(stmt2, tvm.stmt.AttrStmt):
        return _loosely_eq(stmt1, stmt2.body)
    elif isinstance(stmt1, (tvm.expr.Expr, tvm.stmt.Stmt)) and \
         isinstance(stmt2, (tvm.expr.Expr, tvm.stmt.Stmt)):
        for attr in set(dir(stmt1) + dir(stmt2)):
            if attr not in ('node', 'func', 'name'):
                if not (hasattr(stmt1, attr) and hasattr(stmt2, attr)):
                    return False
                else:
                    if not _loosely_eq(getattr(stmt1, attr), getattr(stmt2, attr)):
                        return False
        return True
    else:
        return False

maxby = tvm.comm_reducer(lambda x, y: (tvm.select(x[1] > y[1], x[0], y[0]), tvm.max(x[1], y[1])),
                         lambda t0, t1: (tvm.const(0, t0), tvm.min_value(t1)))

def test_exactly():
    def _check(inp, add_inp, shape, expr1, expr2):
        out = tvm.compute(shape, expr1)
        head = tvm.placeholder(out.shape)
        [jac] = tvm.ir_pass.JacobianRecursive(out, [inp], head)
        ref_jac = tvm.compute(inp.shape, lambda *i: expr2(head, *i))

        s1 = tvm.create_schedule([jac.op])
        mod1 = tvm.lower(s1, [inp, jac, head] + add_inp, simple_mode=True)

        s2 = tvm.create_schedule([ref_jac.op])
        mod2 = tvm.lower(s2, [inp, ref_jac, head] + add_inp, simple_mode=True)

        if not _loosely_eq(mod1, mod2):
            print("=== FAILED FAILED FAILED ===========================================")
        else:
            print("=== PASSED PASSED PASSED ===========================================")
        print("Automatic diff:")
        print(jac.op.body)
        print(mod1)
        print()
        print("Manual diff:")
        print(ref_jac.op.body)
        print(mod2)
        print()
        #raise AssertionError("The gradient computation functions are not symbolically equal")
        print("=============================================================================")
        print("", flush=True)

    X = tvm.placeholder((10, 10, 4), name='X')
    W = tvm.placeholder((3, 3, 4, 5), name='W')

    k = tvm.reduce_axis((0, 3), name="k")
    l = tvm.reduce_axis((0, 3), name="l")
    z = tvm.reduce_axis((0, 4), name="z")

    i = tvm.reduce_axis((0, 7), name="i")
    j = tvm.reduce_axis((0, 7), name="j")
    u = tvm.reduce_axis((0, 4), name="u")

#     _check(W, [X], (7, 7, 5),
           # lambda ii, jj, uu: tvm.sum(X[ii + k, jj + l, z]*W[k, l, z, uu], [k, l, z]),
           # lambda H, kk, ll, zz, uu: tvm.sum(H[i, j, uu]*X[i + kk, j + ll, zz], [i, j]))

    A = tvm.placeholder((10,10), name='A')
    B = tvm.placeholder((10,10), name='B')
    k = tvm.reduce_axis((0, 10), name="k")
    l = tvm.reduce_axis((0, 10), name="l")
    i = tvm.reduce_axis((0, 10), name="i")
    j = tvm.reduce_axis((0, 10), name="j")

  #   _check(A, [B], (10,),
           # lambda ii: tvm.sum(A[ii, k]*B[k, ii], k),
           # lambda H, mm, nn: H[mm]*B[nn, mm])

    # # TODO: Needs transforming Sum(a + b) -> Sum(a) + Sum(b)
    # _check(A, [], (10,),
           # lambda ii: tvm.sum(A[ii, k]*A[k, ii], k),
           # lambda H, mm, nn: H[mm]*A[nn, mm] + H[nn]*A[mm, nn])

    # TODO: Needs some better simplifications
    J = tvm.compute((10,10,10),
                    lambda ii, mm, nn: maxby((tvm.select(tvm.all(tvm.expr.EQ(k, mm),
                                                                 tvm.expr.EQ(ii, nn)),
                                                         B[k, ii], 0.0),
                                              A[k, ii]*B[k, ii]), k))[0]
    _check(A, [B], (10,),
           lambda ii: tvm.max(A[k, ii]*B[k, ii], k),
           lambda H, mm, nn: tvm.sum(H[i]*J[i, mm, nn], i))

    A = tvm.placeholder((10,), name='A')

    # TODO: Needs nonfusion of sums and factoring conditions out
    T = tvm.compute((10,), lambda ii: tvm.sum(B[ii, l], l))
    _check(A, [B], (10, 10),
           lambda ii, jj: tvm.sum(tvm.select(ii == jj, A[k]*B[ii, l], 0.0), [k, l]),
           lambda H, mm: tvm.sum(H[i, i]*T[i], [i]))

if __name__ == "__main__":
    # test_autodiff()
    # test_topi_autodiff()
    # test_some_conv2d_net()
    test_exactly()
