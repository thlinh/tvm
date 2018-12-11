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

def check_equivalence(outputs1, outputs2, inputs, in_range=(-10, 10), iters=10):
    outputs1 = list(outputs1)
    outputs2 = list(outputs2)
    sched1 = tvm.create_schedule([o.op for o in outputs1])
    mout1 = tvm.build(sched1, outputs1 + inputs)

    sched2 = tvm.create_schedule([o.op for o in outputs2])
    mout2 = tvm.build(sched2, outputs2 + inputs)

    arguments1 = [tvm.nd.empty(get_shape(t), t.dtype) for t in outputs1 + inputs]
    arguments2 = [tvm.nd.empty(get_shape(t), t.dtype) for t in outputs1 + inputs]

    for i in range(iters):
        arguments1 = []
        arguments2 = []
        for a in outputs1 + inputs:
            val = np.random.uniform(in_range[0], in_range[1], size=get_shape(a)).astype(a.dtype)
            arguments1.append(tvm.nd.array(val))
            arguments2.append(tvm.nd.array(val))
        mout1(*arguments1)
        mout2(*arguments2)

        for j, _ in enumerate(outputs1):
            tvm.testing.assert_allclose(arguments1[j].asnumpy(), arguments2[j].asnumpy())

# A helper checking the gradient of sum(out) wrt inp
def test_grad(out, inp, args=[], in_range=(-10,10), perf=None):
    if not isinstance(inp, (list, tuple)):
        inp = [inp]

    sout = tvm.create_schedule(out.op)
    mout = tvm.build(sout, [out] + inp + args)

    ones = topi.full_like(out, 1.0)

    t = time.time()
    jacs = list(tvm.differentiate(out, inp, ones))
    print("JAC TIME: ", time.time() - t)

    #  print("=========== Original out ============")
    #  print(tvm.PrintTensorRecursively(out))
    #  print("=========== Jacobians ===============")
    #  for j in jacs: print(tvm.PrintTensorRecursively(j))
    #  print("=====================================")

    t = time.time()
    sjac = tvm.create_schedule([j.op for j in jacs])
    with tvm.build_config(dump_pass_ir=True):
        mjac = tvm.build(sjac, jacs + inp + args)
    print("BUILD TIME: ", time.time() - t)

    lowered = tvm.lower(sjac, jacs + inp + args, simple_mode=True)
    #  print(tvm.ir_pass.Simplify(tvm.ir_pass.CanonicalSimplify(lowered)))
    (iters, mults, mem) = estimate_performance(lowered)
    if perf is None or isinstance(perf, str):
        print("WARNING: No performance information, you may set it to " +
              str((iters, mults, mem)))
        if isinstance(perf, str):
            print("0,/{!r}/{{s/{!r}/{}/}}".format(perf, perf, (iters, mults, mem)))
    elif perf != (iters, mults, mem):
        if iters <= perf[0] and mults <= perf[1] and mem <= perf[2]:
            print("WARNING: Estimated performance {} is better than {}. Use this with sed:"
                  .format((iters, mults, mem), perf))
            print("0,/{}/{{s/{}/{}/}}".format(perf, perf, (iters, mults, mem)))
        elif iters >= perf[0] and mults >= perf[1] and mem >= perf[2]:
            print("WARNING: Estimated performance {} IS WORSE THAN {}"
                  .format((iters, mults, mem), perf))
        else:
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

def test_differentiate_function():
    x = tvm.placeholder((32, 3, 28, 28), name='x')

    w = tvm.placeholder((10, 3, 3, 3), name='w')
    t1 = topi.nn.conv2d(x, w, 1, 0, 1)

    t2 = topi.nn.flatten(t1)
    t3 = topi.sum(t2)

    [dx1, dw1] = tvm.differentiate(t3, [x, w])
    [dx2, dw2] = tvm.differentiate(t2, [x, w], topi.full_like(t2, 1.0))

    check_equivalence([dx1, dw1], [dx2, dw2], [x, w])

    def mydiff(out, inp, head):
        return tvm.compute(inp.shape,
                           lambda ax0, ax1, ax2, ax3: head[ax0, ax3 + ax2*26 + ax1*676])

    res = tvm.differentiate(t3, [x, w], manual={(t2, t1): mydiff})
    check_equivalence(res.result, [dx1, dw1], [x, w])

    res = tvm.differentiate(t3, [x, w], manual={(t2, None): mydiff})
    check_equivalence(res.result, [dx1, dw1], [x, w])

    res = tvm.differentiate(t3, [x, w], manual={(None, t1): mydiff})
    check_equivalence(res.result, [dx1, dw1], [x, w])

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
    test_grad(B, A0, perf=(20100, 142200, 1100))

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
    test_grad(B, A0, perf=(20100, 234200, 2100))

    X = tvm.placeholder((10,), name='X')
    A = tvm.compute((10,), lambda i: X[i] + X[9 - i])
    B = tvm.compute((10,), lambda i: X[i] * X[9 - i])
    Y = topi.tensordot(A, B, 1)
    test_grad(Y, X, perf=(250, 430, 31))

def test_topi_autodiff():
    X = tvm.placeholder((1, 2, 4, 4), name='X')
    W = tvm.placeholder((5, 2, 3, 3), name='W')
    W1 = tvm.placeholder((2, 5, 3, 3), name='W1')
    W2 = tvm.placeholder((1,), name='W1')

    R = topi.nn.conv2d(X, W, 1, 1, 1)
    test_grad(R, [X, W], perf=(3510, 38890, 558))

    R1 = topi.nn.conv2d(topi.nn.relu(R), W1, 1, 0, 1)
    test_grad(R1, [X, W, W1], perf=(8954, 118368, 784))

    R = topi.broadcast_to(W2, (5, 2, 3, 3))
    test_grad(R, [W2], perf=(180, 540, 90))

    R = topi.nn.conv2d(X, topi.broadcast_to(W2, (5, 2, 3, 3)), 1, 1, 1)
    test_grad(R, [X, W2], perf=(3690, 39430, 558))

    R = topi.nn.pool(X, [2, 2], [2, 2], [0, 0, 0, 0], 'avg')
    test_grad(R, X, perf=(40, 400, 8))

    R = topi.nn.pool(X, [2, 2], [2, 2], [0, 0, 0, 0], 'max')
    test_grad(R, X, perf=(168, 4944, 72))

    X = tvm.placeholder((1, 2, 5, 5), name='X')
    W = tvm.placeholder((2, 2, 3, 3), name='W')

    S = topi.reshape(X, (1, 50))
    test_grad(S, [X], perf=(100, 850, 50))

    R = X + topi.nn.conv2d(X + topi.nn.conv2d(X, W, 1, 1, 1), W, 1, 1, 1)
    test_grad(R, [X, W], perf=(7956, 91828, 970))

    S = topi.nn.softmax(topi.reshape(R, (1, 50)))
    test_grad(S, [X, W], perf=(12056, 119121, 1075))

    S = topi.sigmoid(topi.reshape(R, (1, 50)))
    test_grad(S, [X, W], perf=(9106, 113320, 1070))

    S = topi.tanh(topi.reshape(R, (1, 50)))
    test_grad(S, [X, W], perf=(9106, 113320, 1070))

    S = topi.nn.log_softmax(topi.reshape(R, (1, 50)))
    test_grad(S, [X, W], perf=(12006, 118471, 1075))
    test_grad(S, [W], [X], perf=(9992, 93247, 913))

    #  # This is a difficult modular arithmetic case
    #  X = tvm.placeholder((1, 2, 5, 5), name='X')
    #  R = topi.reshape(X, (1, 32))
    #  test_grad(R, [X])

    X = tvm.placeholder((1, 2, 3, 5), name='X')
    Y = tvm.placeholder((1, 2, 7, 5), name='Y')
    S = topi.concatenate((X, Y), 2)
    test_grad(S, [X, Y], perf=(100, 200, 0))

    X = tvm.placeholder((1, 2, 6, 5), name='X')
    (S, R) = topi.split(X, 2, 2)
    test_grad(S, [X], perf=(180, 780, 120))
    test_grad(R, [X], perf=(180, 780, 120))
    R1 = topi.concatenate((S, R), 2)
    test_grad(R1, [X], perf=(420, 1920, 180))
    R2 = topi.concatenate((R, S), 2)
    test_grad(R2, [X], perf=(420, 1920, 180))

def test_stride_dilation():
    X = tvm.placeholder((1, 2, 10, 10), name='X')

    W = tvm.placeholder((2, 2, 1, 1), name='W')

    Y = topi.nn.conv2d(X, W, 1, 0, 1)
    test_grad(Y, [X, W], perf=(1404, 8412, 604))
    Y = topi.nn.conv2d(X, W, 2, 0, 1)
    test_grad(Y, [X, W], perf=(966, 9910, 300))
    Y = topi.nn.conv2d(X, W, 3, 0, 1)
    test_grad(Y, [X, W], perf=(932, 7284, 264))
    Y = topi.nn.conv2d(X, W, 1, 0, 2)
    test_grad(Y, [X, W], perf=(1404, 8412, 604))
    Y = topi.nn.conv2d(X, W, 2, 0, 2)
    test_grad(Y, [X, W], perf=(966, 9910, 300))
    Y = topi.nn.conv2d(X, W, 3, 0, 2)
    test_grad(Y, [X, W], perf=(932, 7284, 264))
    Y = topi.nn.conv2d(X, W, 1, 0, 3)
    test_grad(Y, [X, W], perf=(1404, 8412, 604))
    Y = topi.nn.conv2d(X, W, 2, 0, 3)
    test_grad(Y, [X, W], perf=(966, 9910, 300))
    Y = topi.nn.conv2d(X, W, 3, 0, 3)
    test_grad(Y, [X, W], perf=(932, 7284, 264))

    W = tvm.placeholder((2, 2, 2, 2), name='W')

    Y = topi.nn.conv2d(X, W, 1, 0, 1)
    test_grad(Y, [X, W], perf=(4002, 41780, 1090))
    Y = topi.nn.conv2d(X, W, 2, 0, 1)
    test_grad(Y, [X, W], perf=(1650, 17748, 650))
    Y = topi.nn.conv2d(X, W, 3, 0, 1)
    test_grad(Y, [X, W], perf=(1376, 17336, 632))
    Y = topi.nn.conv2d(X, W, 1, 0, 2)
    test_grad(Y, [X, W], perf=(3660, 55872, 1000))
    Y = topi.nn.conv2d(X, W, 2, 0, 2)
    test_grad(Y, [X, W], perf=(3978, 146240, 1832))
    Y = topi.nn.conv2d(X, W, 3, 0, 2)
    test_grad(Y, [X, W], perf=(3090, 172488, 1112))
    Y = topi.nn.conv2d(X, W, 1, 0, 3)
    test_grad(Y, [X, W], perf=(3378, 52316, 930))
    Y = topi.nn.conv2d(X, W, 2, 0, 3)
    test_grad(Y, [X, W], perf=(3882, 127916, 1826))
    Y = topi.nn.conv2d(X, W, 3, 0, 3)
    test_grad(Y, [X, W], perf=(3834, 126004, 1890))

    W = tvm.placeholder((2, 2, 3, 3), name='W')

    Y = topi.nn.conv2d(X, W, 1, 0, 1)
    test_grad(Y, [X, W], perf=(7580, 80868, 1640))
    Y = topi.nn.conv2d(X, W, 2, 0, 1)
    test_grad(Y, [X, W], perf=(4458, 108812, 1832))
    Y = topi.nn.conv2d(X, W, 3, 0, 1)
    test_grad(Y, [X, W], perf=(1680, 18992, 632))
    Y = topi.nn.conv2d(X, W, 1, 0, 2)
    test_grad(Y, [X, W], perf=(6236, 105692, 1240))
    Y = topi.nn.conv2d(X, W, 2, 0, 2)
    test_grad(Y, [X, W], perf=(8066, 163062, 3980))
    Y = topi.nn.conv2d(X, W, 3, 0, 2)
    test_grad(Y, [X, W], perf=(3752, 353356, 1880))
    Y = topi.nn.conv2d(X, W, 1, 0, 3)
    test_grad(Y, [X, W], perf=(2896, 133764, 520))
    Y = topi.nn.conv2d(X, W, 2, 0, 3)
    test_grad(Y, [X, W], perf=(2586, 52292, 1880))
    Y = topi.nn.conv2d(X, W, 3, 0, 3)
    test_grad(Y, [X, W], perf=(2224, 43644, 280))

    Y = topi.nn.pool(X, [1, 1], [1, 1], [0, 0, 0, 0], 'max')
    test_grad(Y, [X], perf=(200, 400, 0))
    Y = topi.nn.pool(X, [1, 1], [2, 2], [0, 0, 0, 0], 'max')
    test_grad(Y, [X], perf=(450, 3500, 250))
    Y = topi.nn.pool(X, [1, 1], [3, 3], [0, 0, 0, 0], 'max')
    test_grad(Y, [X], perf=(232, 2264, 32))
    Y = topi.nn.pool(X, [2, 2], [1, 1], [0, 0, 0, 0], 'max')
    test_grad(Y, [X], perf=(4242, 139452, 1842))
    Y = topi.nn.pool(X, [2, 2], [2, 2], [0, 0, 0, 0], 'max')
    test_grad(Y, [X], perf=(1050, 30900, 450))
    Y = topi.nn.pool(X, [2, 2], [3, 3], [0, 0, 0, 0], 'max')
    test_grad(Y, [X], perf=(1232, 35328, 432))
    Y = topi.nn.pool(X, [3, 3], [1, 1], [0, 0, 0, 0], 'max')
    test_grad(Y, [X], perf=(18288, 653728, 3888))
    Y = topi.nn.pool(X, [3, 3], [2, 2], [0, 0, 0, 0], 'max')
    test_grad(Y, [X], perf=(8232, 483064, 1632))
    Y = topi.nn.pool(X, [3, 3], [3, 3], [0, 0, 0, 0], 'max')
    test_grad(Y, [X], perf=(2232, 69728, 432))

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
    t = topi.nn.relu(topi.nn.conv2d(t, w1, 1, 0, 1) + topi.reshape(b1, (1, features, 1, 1)))
    t = topi.nn.relu(topi.nn.conv2d(t, w2, 1, 0, 1) + topi.reshape(b2, (1, features, 1, 1)))
    t = topi.nn.pool(t, [2, 2], [2, 2], [0, 0, 0, 0], 'avg')
    t = topi.transpose(t, [0, 2, 3, 1])
    t = topi.nn.flatten(t)
    w3 = tvm.placeholder((dense_units, get_shape(t)[1]))
    t = topi.nn.relu(topi.nn.dense(t, w3, b3))
    t = topi.nn.dense(t, w4, b4)

    t = - topi.sum(y * topi.nn.log_softmax(t)) / batch_size

    weights = [w1, b1, w2, b2, w3, b3, w4, b4]

    test_grad(t, weights, [x, y], in_range=(-1.0, 1.0), perf=(276466, 3393630, 18328))


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
    test_differentiate_function()
    test_autodiff()
    test_topi_autodiff()
    test_stride_dilation()
    test_some_conv2d_net()
