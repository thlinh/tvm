import tvm
from tvm import comm_reducer
from tvm.ir_pass import SimplifyCombiner, Equal

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

if __name__ == "__main__":
    test_simplify_combiner()
