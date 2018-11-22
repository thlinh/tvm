# This example demonstrates Automatic Differentiation for TVM basic operations and TOPI primitives.
# See `test_autodiff()` and `test_nn_autodiff()` for details.

import tvm
import topi
import numpy as np
from nnvm.testing.check_computation import check_numerical_grads
import time

# estimate performance of a stmt. Returns a tuple (loop iterations, multiplications, memory)
def estimate_performance(s):
    from tvm import stmt
    from tvm import expr
    if s is None or isinstance(s, (stmt.AssertStmt, stmt.Free, stmt.Prefetch,
                                   expr.ConstExpr, expr.Var)):
        return (0, 0, 0)
    elif isinstance(s, stmt.Allocate):
        mem = 1
        for e in s.extents:
            mem = e.value * mem
        res = tuple(x + y for x, y in zip(estimate_performance(s.condition),
                                          estimate_performance(s.body)))
        return (res[0], res[1], res[2] + mem)
    elif isinstance(s, stmt.Block):
        return tuple(x + y for x, y in zip(estimate_performance(s.first),
                                           estimate_performance(s.rest)))
    elif isinstance(s, stmt.Evaluate):
        return estimate_performance(s.value)
    elif isinstance(s, stmt.For):
        (iters, mults, mem) = estimate_performance(s.body)
        return (s.extent.value*max(1, iters), s.extent.value*mults, mem)
    elif isinstance(s, stmt.IfThenElse):
        est_body = tuple(max(x, y) for x, y in zip(estimate_performance(s.then_case),
                                                   estimate_performance(s.else_case)))
        return tuple(x + y for x, y in zip(estimate_performance(s.condition), est_body))
    elif isinstance(s, stmt.LetStmt):
        return tuple(x + y for x, y in zip(estimate_performance(s.value),
                                           estimate_performance(s.body)))
    elif isinstance(s, (stmt.ProducerConsumer, stmt.AttrStmt)):
        return estimate_performance(s.body)
    elif isinstance(s, stmt.Provide):
        return estimate_performance(s.value)
    elif isinstance(s, stmt.Realize):
        return tuple(x + y for x, y in zip(estimate_performance(s.condition),
                                           estimate_performance(s.body)))
    elif isinstance(s, stmt.Store):
        return tuple(x + y + z for x, y, z in zip(estimate_performance(s.value),
                                                  estimate_performance(s.index),
                                                  estimate_performance(s.predicate)))
    elif isinstance(s, (expr.Mul, expr.Div, expr.Mod)):
        (iters, mults, mem) = tuple(x + y for x, y in zip(estimate_performance(s.a),
                                                          estimate_performance(s.b)))
        return (iters, mults + 1, mem)
    elif isinstance(s, (expr.BinaryOpExpr, expr.CmpExpr, expr.LogicalExpr)):
        if not hasattr(s, 'b'):
            return estimate_performance(s.a)
        return tuple(x + y for x, y in zip(estimate_performance(s.a),
                                           estimate_performance(s.b)))
    elif isinstance(s, expr.Call):
        res = (0, 0, 0)
        for a in s.args:
            res = tuple(x + y for x, y in zip(estimate_performance(a), res))
        if s.call_type != expr.Call.Halide:
            # expr.If it is a non-halide call (e.g. exp or log), consider it a mul
            res = (res[0], res[1] + 1, res[2])
        return res
    elif isinstance(s, expr.Cast):
        return estimate_performance(s.value)
    elif isinstance(s, expr.Load):
        return tuple(x + y for x, y in zip(estimate_performance(s.index),
                                           estimate_performance(s.predicate)))
    elif isinstance(s, expr.Select):
        return tuple(x + y + z for x, y, z in zip(estimate_performance(s.condition),
                                                  estimate_performance(s.true_value),
                                                  estimate_performance(s.false_value)))

    raise ValueError("Don't know how to estimate performance of {} of type {}"
                     .format(s, type(s)))

def get_shape(tensor):
    return [tvm.ir_pass.Simplify(s).value for s in tensor.shape]

# A helper checking the gradient of sum(out) wrt inp
def test_grad(out, inp, args=[], in_range=(-10,10), perf=None):
    if not isinstance(inp, (list, tuple)):
        inp = [inp]

    sout = tvm.create_schedule(out.op)
    mout = tvm.build(sout, [out] + inp + args)

    ones = topi.full_like(out, 1.0)

    t = time.time()
    jacs = list(tvm.ir_pass.JacobianRecursive(out, inp, ones))
    print("JAC TIME: ", time.time() - t)

    print(tvm.PrintTensorRecursively(jacs[0]))

    t = time.time()
    sjac = tvm.create_schedule([j.op for j in jacs])
    mjac = tvm.build(sjac, jacs + inp + args)
    print("BUILD TIME: ", time.time() - t)

    lowered = tvm.lower(sjac, jacs + inp + args, simple_mode=True)
    (iters, mults, mem) = estimate_performance(lowered)
    if perf is None:
        print("WARNING: No performance information, you may set it to " +
              str((iters, mults, mem)))
    elif perf != (iters, mults, mem):
        print("WARNING: Estimated performance {} does not match {}"
              .format((iters, mults, mem), perf))
        #  if iters > perf[0] or iters < 0.95*perf[0]:
        #      raise AssertionError("The number of iterations {} differ too much from the ref {}"
        #                           .format(iters, perf[0]))
        #  if mem > perf[2] or mem < 0.95*perf[2]:
        #      raise AssertionError("The allocated memory {} differ too much from the ref {}"
        #                           .format(mem, perf[2]))
        #  if mults > perf[1]*1.1 or mults < 0.9*perf[1]:
        #      raise AssertionError("The number of mul ops {} differ too much from the ref {}"
        #                           .format(mults, perf[1]))

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
    test_grad(B, A0, perf=(10100, 40200, 100))

    B = tvm.compute((10, 10), lambda i, j: A0[i, j] + tvm.exp(A0[j, i]), name='B')
    test_grad(B, A0, perf=(10100, 70200, 100))

    B = tvm.compute((10, 10), lambda i, j: tvm.log(tvm.abs(A0[i, j] + tvm.exp(A0[j, i]))), name='B')
    test_grad(B, A0, perf=(10100, 160200, 100))

    B = tvm.compute((10, 10), lambda i, j: tvm.sigmoid(A0[i, j]*A0[i, j]*A0[j, i]), name='B')
    test_grad(B, A0, perf=(10100, 270200, 100))

    B = tvm.compute((10, 10), lambda i, j: tvm.tanh(A0[i, j]*A0[i, j]*A0[j, i]), name='B')
    test_grad(B, A0, perf=(10100, 270200, 100))

    B = tvm.compute((10, 10), lambda i, j: A0[i, j] * A0[j, i], name='B')
    test_grad(B, A0, perf=(10100, 80200, 100))

    B = tvm.compute((10, 10), lambda i, j: tvm.sum(A0[i, k]*A0[k, i] + 5, axis=k), name='B')
    test_grad(B, A0, perf=(110100, 1100200, 10100))

    B = tvm.compute((10, 10), lambda i, j: tvm.max(A0[i, k]*A0[k, j] + 5, axis=k), name='B')
    test_grad(B, A0, perf=(110100, 3430200, 20100))

    B = tvm.compute((10, 10), lambda i, j: A0[i, j] * (A1[j, i] + A0[j, i]), name='B')
    test_grad(B, A0, [A1], perf=(10100, 90200, 100))

    B = tvm.compute((10, 10), lambda i, j: tvm.sum(A0[k, k] - A0[tvm.min(j + k, 9), j]*A0[i, k],
                                                   axis=k),
                    name='B')
    test_grad(B, A0, perf=(110100, 1100200, 10100))

    def fcombine(x, y):
        return x*y

    def fidentity(t0):
        return tvm.const(1, t0)

    prod = tvm.comm_reducer(fcombine, fidentity, name='prod')
    B = tvm.compute((10, 10), lambda i, j: prod(A0[i, k] + A0[k, i], axis=k), name='B')
    test_grad(B, A0, perf=(110100, 2330200, 20100))

def test_topi_autodiff():
    #  X = tvm.placeholder((1, 2, 4, 4), name='X')
    #  W = tvm.placeholder((5, 2, 3, 3), name='W')
    #  W1 = tvm.placeholder((2, 5, 3, 3), name='W1')
    #  W2 = tvm.placeholder((1,), name='W1')

    #  R = topi.nn.conv2d(X, W, 1, 1)
    #  test_grad(R, [X, W], perf=(3542, 39018, 558))

    #  R1 = topi.nn.conv2d(topi.nn.relu(R), W1, 1, 0)
    #  test_grad(R1, [X, W, W1], perf=(8986, 118496, 816))

    #  R = topi.broadcast_to(W2, (5, 2, 3, 3))
    #  test_grad(R, [W2], perf=(180, 540, 91))

    #  R = topi.nn.conv2d(X, topi.broadcast_to(W2, (5, 2, 3, 3)), 1, 1)
    #  test_grad(R, [X, W2], perf=(3754, 39686, 559))

    #  R = topi.nn.pool(X, [2, 2], [2, 2], [0, 0, 0, 0], 'avg')
    #  test_grad(R, X, perf=(168, 1616, 40))

    #  R = topi.nn.pool(X, [2, 2], [2, 2], [0, 0, 0, 0], 'max')
    #  test_grad(R, X, perf=(680, 23632, 264))

    X = tvm.placeholder((1, 2, 5, 5), name='X')
    W = tvm.placeholder((2, 2, 3, 3), name='W')

    S = topi.reshape(X, (1, 50))
    test_grad(S, [X], perf=(1355, 61605, 105))

    R = X + topi.nn.conv2d(X + topi.nn.conv2d(X, W, 1, 1), W, 1, 1)
    test_grad(R, [X, W], perf=(8216, 93908, 1020))

    S = topi.nn.softmax(topi.reshape(R, (1, 50)))
    test_grad(S, [X, W], perf=(60752, 1829334, 2079))

    S = topi.sigmoid(topi.reshape(R, (1, 50)))
    test_grad(S, [X, W], perf=(13330, 351921, 1168))

    S = topi.tanh(topi.reshape(R, (1, 50)))
    test_grad(S, [X, W], perf=(13330, 351921, 1168))

    S = topi.nn.log_softmax(topi.reshape(R, (1, 50)))
    test_grad(S, [X, W], perf=(57923, 2303219, 2092))
    test_grad(S, [W], [X], perf=(33268, 714165, 1049))

    #  # This is a difficult modular arithmetic case
    #  X = tvm.placeholder((1, 2, 5, 5), name='X')
    #  R = topi.reshape(X, (1, 32))
    #  test_grad(R, [X])

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

    test_grad(t, weights, [x, y], in_range=(-1.0, 1.0), perf=(763304, 10288996, 35596))

def test_some_conv2d_net_debug():
    x = tvm.placeholder((1, 1, 14, 14))
    w1 = tvm.placeholder((1, 1, 1, 1))

    t = x
    t = topi.nn.conv2d(t, w1, 1, 0)
    t = x
    t = topi.nn.pool(t, [2, 2], [2, 2], [0, 0, 0, 0], 'avg')

    t = topi.sum(t)

    weights = [w1]

    test_grad(t, [x], [], in_range=(-1.0, 1.0))

# # TODO: Needs transforming Sum(a + b) -> Sum(a) + Sum(b)
# _check(A, [], (10,),
       # lambda ii: tvm.sum(A[ii, k]*A[k, ii], k),
       # lambda H, mm, nn: H[mm]*A[nn, mm] + H[nn]*A[mm, nn])

# TODO: Needs some better simplifications
# J = tvm.compute((10,10,10),
                # lambda ii, mm, nn: maxby((tvm.select(tvm.all(tvm.expr.EQ(k, mm),
                                                             # tvm.expr.EQ(ii, nn)),
                                                     # B[k, ii], 0.0),
                                          # A[k, ii]*B[k, ii]), k))[0]
# _check(A, [B], (10,),
       # lambda ii: tvm.max(A[k, ii]*B[k, ii], k),
       # lambda H, mm, nn: tvm.sum(H[i]*J[i, mm, nn], i))

#  A = tvm.placeholder((10,), name='A')

#  # TODO: Needs nonfusion of sums and factoring conditions out
#  T = tvm.compute((10,), lambda ii: tvm.sum(B[ii, l], l))
#  _check(A, [B], (10, 10),
#         lambda ii, jj: tvm.sum(tvm.select(ii == jj, A[k]*B[ii, l], 0.0), [k, l]),
#         lambda H, mm: tvm.sum(H[i, i]*T[i], [i]))

if __name__ == "__main__":
    test_some_conv2d_net_debug()
    #  test_autodiff()
    #  test_topi_autodiff()
    #  test_some_conv2d_net()
    #  test_exactly()
