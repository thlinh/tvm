
# TVM-level automatic differentiation
This notebook shows how to use tvm-level automatic differentiation and discusses how it works internally, what you can expect to be differentiated well, and what still requires some more work. Note that this is a work-in-progress and the result of differentiating certain operations is not as performant yet as we want it to be.

Let's start by importing modules and defining some helpers.


```python
import tvm
import topi
import time
import math
import numpy as np

def get_shape(tensor):
    return [tvm.ir_pass.Simplify(s).value for s in tensor.shape]

# This function builds a tvm function, runs it for several iterations, 
# and returns a string with the median time and some additional statistics
def measure_performance(outputs, inputs, min_seconds=1):
    sched = tvm.create_schedule([o.op for o in outputs])
    mout = tvm.build(sched, outputs + inputs)
    
    arguments = [tvm.nd.empty(get_shape(t), t.dtype) for t in outputs + inputs]
    
    times = []
    while sum(times) < min_seconds:
        before = time.time()
        mout(*arguments)
        after = time.time()
        times.append(after - before)
        
    return "{}ms median, avg={}, std={} ({} iters)".format(
        int(1000*np.median(times)), int(1000*np.mean(times)), int(1000*np.std(times)), len(times))

# Print the lowered representation
def show_lowered(outputs, inputs):
    sout = tvm.create_schedule([o.op for o in outputs])
    mout = tvm.lower(sout, outputs + inputs, simple_mode=True)
    print(mout)
    
    
```

## How to use automatic differentiation
Basically, all you need is the function `tvm.differentiate` which takes a tensor, differentiates it with respect to other given tensors using reverse accumulation, and applies certain optimizations. Let's consider an example: 


```python
# inputs
X = tvm.placeholder((32, 10000), name='X')
W = tvm.placeholder((3000, 10000), name='W')
B = tvm.placeholder((3000,), name='B')

# output
Y = topi.nn.dense(X, W, B)

# Adjoint (head gradients). In this case it is has the same shape as Y and
# represents the gradient of some hypothetical scalar loss with respect to Y.
# In the most common case Y will be the loss itself with the shape (1,)
# and H will simply be a scalar 1, but here we want to look at a more general case.
H = tvm.placeholder(Y.shape, name='H')

# Get Jacobians of Y wrt W and B, multiplied by H,
# in other words, get gradients of some loss wrt W and B
# given H, the gradient of this loss wrt Y
[dW, dB] = tvm.differentiate(Y, [W, B], H)
```


```python
print("forward ", measure_performance([Y], [X, B, W]))
print("backward", measure_performance([dW, dB], [X, B, W, H]))
```

    forward  863ms median, avg=863, std=28 (2 iters)
    backward 566ms median, avg=566, std=31 (2 iters)


Note that `H` may be omitted, and then the Jacobian itself will be returned (which is called gradient when Y is a scalar). 


```python
loss = topi.sum(Y)
res = tvm.differentiate(loss, [W, B])
[dW, dB] = res
```

The result `res` mimics a list, but it also contains a couple of additional fields, namely `adjoints` and `adjoint_summands`. The `adjoints` dict maps original tensors to corresponding adjoints (gradients of the output with respect to this particular tensor):


```python
res.adjoints
```




    {Tensor(shape=[3000, 10000], op.name=W): Tensor(shape=[3000, 10000], op.name=compute.W.grad),
     Tensor(shape=[3000], op.name=B): Tensor(shape=[3000], op.name=compute.B.grad),
     Tensor(shape=[32, 3000], op.name=compute): Tensor(shape=[32, 3000], op.name=compute_red.compute.grad),
     Tensor(shape=[32, 3000], op.name=compute): Tensor(shape=[32, 3000], op.name=compute.compute.grad),
     Tensor(shape=[], op.name=compute_red): Tensor(shape=[], op.name=identity)}



Each adjoint may be a sum of several components when there are several dependency paths from the output to the tensor. To access each component there is a dict called `adjoint_summands`:


```python
X = tvm.placeholder((10,), name='X')
A = tvm.compute((10,), lambda i: X[i] + X[9 - i])
B = tvm.compute((10,), lambda i: X[i] * X[9 - i])
Y = topi.tensordot(A, B, 1)

# If we don't specify the inputs, it will differentiate
# with respect to all tensors
res = tvm.differentiate(Y)

print("The adjoint of X is a sum:")
print(res.adjoints[X].op.body[0])

print("\nThe component that came from A:")
print(res.adjoint_summands[X][A].op.body[0])

print("\nThe component that came from B:")
print(res.adjoint_summands[X][B].op.body[0])
```

    The adjoint of X is a sum:
    (compute.X.grad(ax0) + compute.X.grad(ax0))
    
    The component that came from A:
    reduce(combiner=comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0.000000f]), source=[(tensor.compute.grad(k0)*(select((ax0 == k0), 1.000000f, 0.000000f) + select(((ax0 + k0) == 9), 1.000000f, 0.000000f)))], axis=[iter_var(k0, Range(min=0, extent=10))], where=(uint1)1, value_index=0)
    
    The component that came from B:
    reduce(combiner=comm_reducer(result=[(x + y)], lhs=[x], rhs=[y], identity_element=[0.000000f]), source=[(tensor.compute.grad(k0)*(select((ax0 == k0), X((9 - k0)), 0.000000f) + select(((k0 + ax0) == 9), X(k0), 0.000000f)))], axis=[iter_var(k0, Range(min=0, extent=10))], where=(uint1)1, value_index=0)


`adjoints` and `adjoint_summands` may probably be useful when manually scheduling the resulting computation graph.

## How it works internally
Internally `tvm.differentiate` recursively builds adjoints using the `tvm.autodiff.DiffBuildingBlock` function. This function takes three parameters: `output`, `input` and `head`. `output` is some tensor for which the adjoint has already been computed, and `head` is its adjoint. `input` is a tensor for which we want to compute the adjoint, and which is used from within the compute body of `output`. The function returns something close to `tensordot(head, Jacobian())` (but heavily optimized) where `Jacobian(Y, W)` simply differentiates `Y` wrt `W` assuming that `Y` depens on `W` only directly. So let's look at the `Jacobian` function. It has additional parameter which indicates whether to perform optimizations, so let's look at an unoptimized result of this function.


```python
X = tvm.placeholder((32, 10000), name='X')
W = tvm.placeholder((3000, 10000), name='W')

Y = topi.nn.dense(X, W)
dYdW = tvm.autodiff.Jacobian(Y, W, False)

# This function prints out a tensor with all its dependencies in a slightly more readable
# format, in particular, it prints every attribute of a reduction on a new line
print("The origiginal tensor Y:")
print(tvm.PrintTensorRecursively(Y))
print("\nJacobian(Y, W):")
print(tvm.PrintTensorRecursively(dYdW))
```

    The origiginal tensor Y:
    tensor compute{0xb92eb0}[0] : float32 [32, 3000]
    axes (i : [0, 31], j : [0, 2999])
    Reduction
        identity [0.000000f]
        lhs [x]  rhs [y]
        combiner [(x + y)]
        axes (k : [0, 9999])
        condition (uint1)1
        source[0] = (X(i, k)*W(j, k))
    
    tensor X{0x1dde3a0}[0] : float32 [32, 10000]
        placeholder(X, 0x1dde3a0)
    
    tensor W{0x1ca8660}[0] : float32 [3000, 10000]
        placeholder(W, 0x1ca8660)
    
    
    
    Jacobian(Y, W):
    tensor compute.jacobian{0x165b360}[0] : float32 [32, 3000, 3000, 10000]
    axes (i : [0, 31], j : [0, 2999], jac_i0 : [0, 2999], jac_i1 : [0, 9999])
    Reduction
        identity [0.000000f]
        lhs [x.der]  rhs [y.der]
        combiner [(x.der + y.der)]
        axes (k : [0, 9999])
        condition (uint1)1
        source[0] = (X(i, k)*float32(((jac_i0 == j) && (jac_i1 == k))))
    
    tensor X{0x1dde3a0}[0] : float32 [32, 10000]
        placeholder(X, 0x1dde3a0)
    
    


You can see that `W(j, k)` in the original tensor Y became `float32(((jac_i0 == j) && (jac_i1 == k)))` in the Jacobian, which is the derivative of `W(j, k)` wrt `W(jac_i0, jac_i1)` (it's equal to 1 if the corresponding indices coincide, otherwise it's zero). Of course, computing this Jacobian is very inefficient, because it consists of summing over mostly zero values, so it should be optimized by propagating the information that `jac_i1 == k` and completely removing the summation. It may be done with the function `OptimizeAndLiftNonzeronessConditions` (which is called by the `Jacobian` function by default). Let's call it manually:


```python
dYdW_optimized = tvm.ir_pass.OptimizeAndLiftNonzeronessConditions(dYdW)
print(tvm.PrintTensorRecursively(dYdW_optimized))
```

    tensor compute.jacobian{0x1996200}[0] : float32 [32, 3000, 3000, 10000]
    axes (i : [0, 31], j : [0, 2999], jac_i0 : [0, 2999], jac_i1 : [0, 9999])
        select((j == jac_i0), tensor(jac_i1, i), 0.000000f)
    
    tensor tensor{0x201c2c0}[0] : float32 [10000, 32]
    axes (jac_i1 : [0, 9999], i : [0, 31])
        X(i, jac_i1)
    
    tensor X{0x1dde3a0}[0] : float32 [32, 10000]
        placeholder(X, 0x1dde3a0)
    
    


The reduction was eliminated completely, and replaced with a conditional expression returning `X(i, jac_i1)` if `j == jac_i0` and 0 otherwise.

The condition `j == jac_i0` may be used to eliminate another reduction. Recall that the Jacobian is used in a formula looking similar to `tensordot(H, Jacobian(Y, W))`, so the reduction to be eliminated is a summation used in matrix multiplication. To perform this transformation, `JacobianRecursive` inlines the Jacobian and calls `OptimizeAndLiftNonzeronessConditions` once more. Let's do this manually:


```python
# Generalized matmul works with tensors of arbitrary dimensions and takes
# an additional parameter: the number of dimensions to contract. It is
# semantically equivalent to reshaping into two matrices, 
# performing matrix multiplication, and then reshaping back
dLdW = topi.tensordot(H, dYdW_optimized, 2)

# We have to inline dYdW_optimized because OptimizeAndLiftNonzeronessConditions works
# only with a single tensor
dLdW_inlined = tvm.ir_pass.InlineNonReductions(dLdW, [dYdW_optimized])

# Perform the main optimization
dLdW_optimized = tvm.ir_pass.OptimizeAndLiftNonzeronessConditions(dLdW_inlined)
print(tvm.PrintTensorRecursively(dLdW_optimized))
```

    tensor tensor{0x1fbbc90}[0] : float32 [3000, 10000]
    axes (ax0 : [0, 2999], ax1 : [0, 9999])
        extracted(ax0, ax1)
    
    tensor extracted{0x1fd3e60}[0] : float32 [3000, 10000]
    axes (ax0 : [0, 2999], ax1 : [0, 9999])
    Reduction
        identity [0.000000f]
        lhs [x]  rhs [y]
        combiner [(x + y)]
        axes (k0 : [0, 31])
        condition (uint1)1
        source[0] = (H(k0, ax0)*tensor(ax1, k0))
    
    tensor H{0x1d758f0}[0] : float32 [32, 3000]
        placeholder(H, 0x1d758f0)
    
    tensor tensor{0x201c2c0}[0] : float32 [10000, 32]
    axes (jac_i1 : [0, 9999], i : [0, 31])
        X(i, jac_i1)
    
    tensor X{0x1dde3a0}[0] : float32 [32, 10000]
        placeholder(X, 0x1dde3a0)
    
    


You can see that now there is only one reduction axis left, an there no comparison `j == jac_i0` anymore.

## Providing some gradients manually
Sometimes automatic differentiation does poor job and it may be desired to provide a custom differentiation function for certain tensors. By default `tvm.differentiate` uses the function `tvm.autodiff.DiffBuildingBlock`, but it is possible to provide a custom function using the `fdiff` parameter. It is also possible to change the differentiation function only for some specific cases using the `manual` parameter:


```python
x = tvm.placeholder((32, 3, 28, 28), name='x')
w1 = tvm.placeholder((10, 3, 3, 3), name='w1')
t1 = topi.nn.conv2d(x, w1, 1, 0, 1)
t2 = topi.nn.flatten(t1)
t3 = topi.sum(t2)
t3

# Currently flatten is differentiated quite well, but there is an
# annoying big condition that cannot be simplified
res = tvm.differentiate(t3)
print("======= Autodiff couldn't simplify some expressions:")
print(tvm.PrintTensorRecursively(res.adjoints[t1]))

def mydiff(out, inp, head):
    return tvm.compute(inp.shape, lambda ax0, ax1, ax2, ax3: head[ax0, ax3 + ax2*26 + ax1*676])

# Here we provide the better version manually by specifying that
# for the pair of tensors (t2, t1), t2 being the output, and t1 the input,
# the function `mydiff` should be used instead of DiffBuildingBlock
print("======= We can simplify them manually:")
res = tvm.differentiate(t3, [x, w1], manual={(t2, t1): mydiff})
print(tvm.PrintTensorRecursively(res.adjoints[t1]))
```

    ======= Autodiff couldn't simplify some expressions:
    tensor compute.compute.grad{0x1496e40}[0] : float32 [32, 10, 26, 26]
    axes (ax0 : [0, 31], ax1 : [0, 9], ax2 : [0, 25], ax3 : [0, 25])
        select(((((((((((((ax1*676) - (ax2*26)) - (select((0 < (((ax1*676) - ax3) - (ax2*26))), ax1, (ax1 + (((0 - ax3) - (ax2*26))/676)))*676)) <= ax3) && ((((ax3 + (ax2*26)) + (select((0 < (((ax1*676) - ax3) - (ax2*26))), ax1, (ax1 + (((0 - ax3) - (ax2*26))/676)))*676)) - (ax1*676)) <= 675)) && (((ax1*676) - ax3) <= ((ax2*26) + (select((0 < (((ax1*676) - ax3) - (ax2*26))), ax1, (ax1 + (((0 - ax3) - (ax2*26))/676)))*676)))) && ((((ax3 + (ax2*26)) + (select((0 < (((ax1*676) - ax3) - (ax2*26))), ax1, (ax1 + (((0 - ax3) - (ax2*26))/676)))*676)) - 675) <= (ax1*676))) && ((((ax1*676) - (ax2*26)) - (select((0 < (((ax1*676) - ax3) - (ax2*26))), ax1, (ax1 + (((0 - ax3) - (ax2*26))/676)))*676)) <= ax3)) && ((((ax3 + (ax2*26)) + (select((0 < (((ax1*676) - ax3) - (ax2*26))), ax1, (ax1 + (((0 - ax3) - (ax2*26))/676)))*676)) - 675) <= (ax1*676))) && ((((ax1*676) - ax3) - (ax2*26)) <= (select((0 < (((ax1*676) - ax3) - (ax2*26))), ax1, (ax1 + (((0 - ax3) - (ax2*26))/676)))*676))) && (0 <= select((0 < (((ax1*676) - ax3) - (ax2*26))), ax1, (ax1 + (((0 - ax3) - (ax2*26))/676))))) && ((((ax3 + (ax2*26)) + (select((0 < (((ax1*676) - ax3) - (ax2*26))), ax1, (ax1 + (((0 - ax3) - (ax2*26))/676)))*676)) - 675) <= (ax1*676))), compute_red.compute.grad(ax0, ((ax3 + (ax2*26)) + (select((0 < (((ax1*676) - ax3) - (ax2*26))), ax1, (ax1 + (((0 - ax3) - (ax2*26))/676)))*676))), 0.000000f)
    
    tensor compute_red.compute.grad{0x1f32340}[0] : float32 [32, 6760]
    axes (ax0 : [0, 31], ax1 : [0, 6759])
        float32((uint1)1)
    
    
    ======= We can simplify them manually:
    tensor compute{0x1e67790}[0] : float32 [32, 10, 26, 26]
    axes (ax0 : [0, 31], ax1 : [0, 9], ax2 : [0, 25], ax3 : [0, 25])
        compute_red.compute.grad(ax0, ((ax3 + (ax2*26)) + (ax1*676)))
    
    tensor compute_red.compute.grad{0x1dd3730}[0] : float32 [32, 6760]
    axes (ax0 : [0, 31], ax1 : [0, 6759])
        float32((uint1)1)
    
    


# Supported operations
Here is a list of operations which seem to be differentiated quite well by our autodiff.

## Dense


```python
X = tvm.placeholder((32, 1000), name='X')
W = tvm.placeholder((1000, 1000), name='W')
B = tvm.placeholder((1000,), name='B')

Y = topi.nn.dense(X, W, B)

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.differentiate(Y, [X, W, B], H)
```


```python
print("forward ", measure_performance([Y], [X, B, W]))
print("backward", measure_performance(list(grads), [X, B, W, H]))
```

    forward  26ms median, avg=27, std=5 (36 iters)
    backward 54ms median, avg=57, std=11 (18 iters)


## Conv2D
with `dilation=1`


```python
X = tvm.placeholder((32, 17, 28, 28), name='X')
W = tvm.placeholder((19, 17, 3, 3), name='W')
Y = topi.nn.conv2d(X, W, [1, 1], [0, 0], 1)

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.differentiate(Y, [X, W], H)
```


```python
print("forward ", measure_performance([Y], [X, W]))
print("backward", measure_performance(list(grads), [X, W, H]))
```

    forward  58ms median, avg=59, std=9 (17 iters)
    backward 102ms median, avg=109, std=19 (10 iters)



```python
X = tvm.placeholder((32, 17, 58, 58), name='X')
W = tvm.placeholder((19, 17, 5, 5), name='W')
Y = topi.nn.conv2d(X, W, [3, 3], [1, 1], 1)

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.differentiate(Y, [X, W], H)

print("forward ", measure_performance([Y], [X, W]))
print("backward", measure_performance(list(grads), [X, W, H]))
```

    forward  78ms median, avg=80, std=5 (13 iters)
    backward 172ms median, avg=191, std=42 (6 iters)


# Somewhat supported

## Average pooling
The performance is suspicious but the generated code looks ok except for large if expressions which cannot be eliminated by subsequent passes (the problem in not nearly as horrible as with max pooling).


```python
X = tvm.placeholder((32, 17, 280, 280), name='X')
Y = topi.nn.pool(X, [2, 2], [2, 2], [0, 0, 0, 0], 'avg')

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.differentiate(Y, [X], H)
```


```python
print("forward ", measure_performance([Y], [X]))
print("backward", measure_performance(list(grads), [X, H]))
```

    forward  13ms median, avg=14, std=2 (72 iters)
    backward 94ms median, avg=98, std=14 (11 iters)



```python
print(tvm.PrintTensorRecursively(grads[0]))
```

    tensor tensor.X.grad{0x22c5180}[0] : float32 [32, 17, 280, 280]
    axes (ax0 : [0, 31], ax1 : [0, 16], ax2 : [0, 279], ax3 : [0, 279])
        select((((((((ax2 - 1) <= (select((1 < ax2), (ax2/2), ((ax2 - 1)/2))*2)) && (0 <= select((1 < ax2), (ax2/2), ((ax2 - 1)/2)))) && ((select((1 < ax2), (ax2/2), ((ax2 - 1)/2))*2) <= ax2)) && ((ax3 - 1) <= (select((1 < ax3), (ax3/2), ((ax3 - 1)/2))*2))) && (0 <= select((1 < ax3), (ax3/2), ((ax3 - 1)/2)))) && ((select((1 < ax3), (ax3/2), ((ax3 - 1)/2))*2) <= ax3)), (H(ax0, ax1, select((1 < ax2), (ax2/2), ((ax2 - 1)/2)), select((1 < ax3), (ax3/2), ((ax3 - 1)/2)))*0.250000f), 0.000000f)
    
    tensor H{0x23c26e0}[0] : float32 [32, 17, 140, 140]
        placeholder(H, 0x23c26e0)
    
    


## Softmax (homebrewn)
Softmax from topi causes performance problems (see below), but we can write our own softmax which works better but still not perfectly (seems like some performance problems when used after other layers like dense).


```python
X = tvm.placeholder((60, 100), name="X")
W = tvm.placeholder((1000, 1000), name='W')

exps = topi.exp(topi.nn.dense(X, W))
sumexps = topi.sum(exps, axis=-1, keepdims=True)
Y = exps/sumexps

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.differentiate(Y, [X, W], H)
```


```python
print("forward ", measure_performance([Y], [X, W]))
print("backward", measure_performance(list(grads), [X, W, H]))
```

    forward  6ms median, avg=6, std=2 (154 iters)
    backward 18ms median, avg=19, std=4 (53 iters)


## Flatten
Flatten uses the division and modulo operations. Recently we've implemented a transformation to deal with them, so this operation is pretty much supported, although the resulting code is not perfect (it contains a huge select which should be simplified out in theory).


```python
X = tvm.placeholder((32, 100, 20, 25), name='X')
Y = topi.nn.flatten(X)

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.differentiate(Y, [X], H)
```


```python
print("forward ", measure_performance([Y], [X]))
print("backward", measure_performance(list(grads), [X, H]))
```

    forward  5ms median, avg=6, std=2 (165 iters)
    backward 7ms median, avg=8, std=3 (123 iters)



```python
print(tvm.PrintTensorRecursively(grads[0]))
```

    tensor compute.X.grad{0x2071170}[0] : float32 [32, 100, 20, 25]
    axes (ax0 : [0, 31], ax1 : [0, 99], ax2 : [0, 19], ax3 : [0, 24])
        select(((((((((((((ax1*500) - (ax2*25)) - (select((0 < (((ax1*500) - ax3) - (ax2*25))), ax1, (ax1 + (((0 - ax3) - (ax2*25))/500)))*500)) <= ax3) && ((((ax3 + (ax2*25)) + (select((0 < (((ax1*500) - ax3) - (ax2*25))), ax1, (ax1 + (((0 - ax3) - (ax2*25))/500)))*500)) - (ax1*500)) <= 499)) && (((ax1*500) - ax3) <= ((ax2*25) + (select((0 < (((ax1*500) - ax3) - (ax2*25))), ax1, (ax1 + (((0 - ax3) - (ax2*25))/500)))*500)))) && ((((ax3 + (ax2*25)) + (select((0 < (((ax1*500) - ax3) - (ax2*25))), ax1, (ax1 + (((0 - ax3) - (ax2*25))/500)))*500)) - 499) <= (ax1*500))) && ((((ax1*500) - (ax2*25)) - (select((0 < (((ax1*500) - ax3) - (ax2*25))), ax1, (ax1 + (((0 - ax3) - (ax2*25))/500)))*500)) <= ax3)) && ((((ax3 + (ax2*25)) + (select((0 < (((ax1*500) - ax3) - (ax2*25))), ax1, (ax1 + (((0 - ax3) - (ax2*25))/500)))*500)) - 499) <= (ax1*500))) && ((((ax1*500) - ax3) - (ax2*25)) <= (select((0 < (((ax1*500) - ax3) - (ax2*25))), ax1, (ax1 + (((0 - ax3) - (ax2*25))/500)))*500))) && (0 <= select((0 < (((ax1*500) - ax3) - (ax2*25))), ax1, (ax1 + (((0 - ax3) - (ax2*25))/500))))) && ((((ax3 + (ax2*25)) + (select((0 < (((ax1*500) - ax3) - (ax2*25))), ax1, (ax1 + (((0 - ax3) - (ax2*25))/500)))*500)) - 499) <= (ax1*500))), H(ax0, ((ax3 + (ax2*25)) + (select((0 < (((ax1*500) - ax3) - (ax2*25))), ax1, (ax1 + (((0 - ax3) - (ax2*25))/500)))*500))), 0.000000f)
    
    tensor H{0x221e0a0}[0] : float32 [32, 50000]
        placeholder(H, 0x221e0a0)
    
    


Here the compiler has to figure out that `k1.shifted` is directly expressible using `ax1, ax2, ax3`.

## Max pooling
Reducing with other combiners, like max, is a bit trickier than summation. We used to have a problem with memory allocation: differentiating max pooling resulted in tesors larger than 2^31, but the problem is fixed now. Still, it's not very performant, the reason is yet unknown.


```python
X = tvm.placeholder((32, 64, 100, 100), name='X')
Y = topi.nn.pool(X, [2, 2], [2, 2], [0, 0, 0, 0], 'max')

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.differentiate(Y, [X], H)
```


```python
print("forward ", measure_performance([Y], [X]))
print("backward", measure_performance(list(grads), [X, H]))
```

    forward  7ms median, avg=7, std=1 (135 iters)
    backward 765ms median, avg=765, std=7 (2 iters)


## Softmax
Softmax uses max behind the scenes, causing some performance problems. We havent't yet investingated into it though.


```python
X = tvm.placeholder((60, 100), name="X")
W = tvm.placeholder((1000, 1000), name='W')

Y = topi.nn.softmax(topi.nn.dense(X, W))

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.differentiate(Y, [X, W], H)
```


```python
print("forward ", measure_performance([Y], [X, W]))
print("backward", measure_performance(list(grads), [X, W, H]))
```

    forward  7ms median, avg=8, std=2 (122 iters)
    backward 183ms median, avg=187, std=28 (6 iters)


# Unsupported
Currently **dilated convolutions are not supported** (wrong gradients, probably a simplification bug or something like this).
