import numpy as np
import tvm
from tvm import comm_reducer
from tvm.ir_pass import SimplifyCombiner, FuseTensors, Equal

def get_shape(tensor):
    return [s.value for s in tensor.shape]

def check_eq(t1, t2, args):
    print(t1.op.body)
    print(t2.op.body)

    s1 = tvm.create_schedule(t1.op)
    m1 = tvm.build(s1, [t1] + args)

    s2 = tvm.create_schedule(t2.op)
    m2 = tvm.build(s2, [t2] + args)

    def fun(args):
        aaa = [tvm.nd.empty(get_shape(t1), out.dtype)] + \
            [tvm.ndarray.array(kwargs[a.name]) for a in [inp] + args]
        mout(*aaa)
        return aaa[0].asnumpy().sum()

    for _ in range(5):
        arg_vals = [tvm.ndarray.array(np.random.uniform(-10, 10, size=get_shape(a)).astype(a.dtype))
                    for a in [t1] + args]
        m1(*arg_vals)
        res1 = arg_vals[0].asnumpy()
        m2(*arg_vals)
        res2 = arg_vals[0].asnumpy()

        np.testing.assert_allclose(res1, res2, atol=1e-3, rtol=1e-2)

def test_simplify_combiner():
    prod = comm_reducer(lambda x, y: x*y, lambda t0: tvm.const(1, t0))

    sum_and_prod = comm_reducer(lambda x, y: (x[0] + y[0],
                                              x[1]*y[1]),
                                lambda t0, t1: (tvm.const(0, t0),
                                                tvm.const(5, t0) - tvm.const(4, t0)))

    sum_and_prod2 = comm_reducer(lambda x, y: (x[0] + y[0],
                                               x[1]*y[1] + 0*x[0] + y[0] - y[0]),
                                 lambda t0, t1: (tvm.const(5, t0) - tvm.const(5, t0),
                                                 tvm.const(1, t1)))

    some_reducer1 = comm_reducer(lambda x, y: (x[0] + y[0],
                                               x[0] + y[0] + x[1] + y[1],
                                               x[0]*y[2] + y[0]*x[2],
                                               x[1] + y[2],
                                               4.0),
                                 lambda t0, t1, t2, t3, t4: (tvm.const(0, t0),
                                                             tvm.const(1, t1),
                                                             tvm.const(2, t2),
                                                             tvm.const(3, t3),
                                                             tvm.const(4, t4)))

    k = tvm.reduce_axis((0, 10), name="k")
    A = tvm.placeholder((10,), name='A')

    assert Equal(SimplifyCombiner(sum_and_prod((A[k], A[10-k]), k)[0]), tvm.sum(A[k], k))
    assert Equal(SimplifyCombiner(sum_and_prod((A[k], A[10-k]), k)[1]), prod(A[10-k], k))
    assert Equal(SimplifyCombiner(sum_and_prod((A[k], A[10-k]), k)[0], False),
                 sum_and_prod((A[k], A[10-k]), k)[0])

    assert Equal(SimplifyCombiner(sum_and_prod2((A[k], A[10-k]), k)[0]), tvm.sum(A[k], k))
    assert Equal(SimplifyCombiner(sum_and_prod2((A[k], A[10-k]), k)[1]), prod(A[10-k], k))

    assert [len(SimplifyCombiner(some_reducer1((A[k], A[10-k], A[0], A[k+1], A[k]), k)[j]).source)
            for j in range(5)] == [1, 2, 2, 4, 1]

def test_fuse_tensors():
    k = tvm.reduce_axis((0, 5), name="k")
    l = tvm.reduce_axis((0, 5), name="l")
    A = tvm.placeholder((10,), name='A')

    B = tvm.compute((10,), lambda i: tvm.sum(A[(k + i) % 10], k))
    C = tvm.compute((10,10), lambda i, j: B[i]*B[j])
    F = FuseTensors(C, [B])
    assert isinstance(F.op.body[0], tvm.expr.Reduce)
    check_eq(C, F, [A])

    B = tvm.compute((10,), lambda i: tvm.sum(A[(k + i) % 10], k, where=((k + i) % 2 == 0)))
    C = tvm.compute((10,10), lambda i, j: B[i]*B[j])
    F = FuseTensors(C, [B])
    assert isinstance(F.op.body[0], tvm.expr.Reduce)
    check_eq(C, F, [A])

    B = tvm.compute((10,), lambda i: tvm.sum(A[(k + i) % 10], k))
    C = tvm.compute((10,10), lambda i, j: tvm.sum(B[(l + i) % 10]*B[(l + j) % 10], l))
    F = FuseTensors(C, [B])
    assert isinstance(F.op.body[0], tvm.expr.Reduce)
    check_eq(C, F, [A])

    B = tvm.compute((10,), lambda i: tvm.max(A[(k + i) % 10], k))
    C = tvm.compute((10,10), lambda i, j: tvm.sum(B[(l + i) % 10]*B[(l + j) % 10], l))
    F = FuseTensors(C, [B])
    check_eq(C, F, [A])

    B = tvm.compute((10,), lambda i: tvm.sum(A[(k + i) % 10], k))
    C = tvm.compute((10,10), lambda i, j: B[i]*5)
    F = FuseTensors(C, [B])
    assert isinstance(F.op.body[0], tvm.expr.Reduce)
    check_eq(C, F, [A])

    B = tvm.compute((10,), lambda i: tvm.sum(A[(k + i) % 10], k))
    C = tvm.compute((10,10), lambda i, j: 5*B[i])
    F = FuseTensors(C, [B])
    assert isinstance(F.op.body[0], tvm.expr.Reduce)
    check_eq(C, F, [A])

    B = tvm.compute((10,), lambda i: tvm.sum(A[(k + i) % 10], k))
    C = tvm.compute((10,10), lambda i, j: B[i] + B[j])
    F = FuseTensors(C, [B])
    # not yet
    assert not isinstance(F.op.body[0], tvm.expr.Reduce)
    check_eq(C, F, [A])

    X = tvm.compute((10,), lambda i: tvm.sum(A[(l + i*2) % 10], l))
    B = tvm.compute((10,), lambda i: tvm.sum(X[(k + i) % 10], k))
    C = tvm.compute((10,10), lambda i, j: X[i]*B[j])
    F = FuseTensors(C, [])
    assert isinstance(F.op.body[0], tvm.expr.Reduce)
    check_eq(C, F, [A])

if __name__ == "__main__":
    #test_simplify_combiner()
    test_fuse_tensors()
