
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
# and returns the time of running one iteration in milliseconds
def measure_performance(outputs, inputs, min_seconds=1):
    sched = tvm.create_schedule([o.op for o in outputs])
    mout = tvm.build(sched, outputs + inputs)
    
    arguments = [tvm.nd.empty(get_shape(t), t.dtype) for t in outputs + inputs]
    
    seconds = 0
    iters = 0
    iters_to_do = 1
    while seconds < min_seconds:
        if seconds > 0:
            iters_to_do = min(100, math.ceil(iters / seconds))
        before = time.time()
        for i in range(iters_to_do):
            mout(*arguments)
        after = time.time()
        seconds += after - before
        iters += iters_to_do
        
    return int(1000*(seconds/iters))

# Print the lowered representation
def show_lowered(outputs, inputs):
    sout = tvm.create_schedule([o.op for o in outputs])
    mout = tvm.lower(sout, outputs + inputs, simple_mode=True)
    print(mout)
    
    
```

## How to use automatic differentiation
Basically, all you need is the function `tvm.ir_pass.JacobianRecursive` which takes a tensor, differentiates it with respect to other given tensors using reverse accumulation, and applies certain optimizations. Let's consider an example: 


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
[dW, dB] = tvm.ir_pass.JacobianRecursive(Y, [W, B], H)
```


```python
print("forward ", measure_performance([Y], [X, B, W]))
print("backward", measure_performance([dW, dB], [X, B, W, H]))
```

    forward  858
    backward 928


## How it works internally
Internally `JacobianRecursive` builds tensors of the form close to `matmul(H, Jacobian(Y, W))` where `Jacobian(Y, W)` simply differentiates Y wrt W assuming that Y directly uses W (`JacobianRecursive` doesn't make this assumption). So let's look at the `Jacobian` function. It has additional parameter which indicates whether to perform optimizations, so let's look at an unoptimized result of this function.


```python
Y = topi.nn.dense(X, W)
dYdW = tvm.ir_pass.Jacobian(Y, W, False)

# This function prints out a tensor with all its dependencies in a slightly more readable
# format, in particular, it prints every attribute of a reduction on a new line
print("The origiginal tensor Y:")
print(tvm.PrintTensorRecursively(Y))
print("\nJacobian(Y, W):")
print(tvm.PrintTensorRecursively(dYdW))
```

    The origiginal tensor Y:
    tensor compute{0x271c110}[0] : float32 [32, 3000]
    axes (i : [0, 31], j : [0, 2999])
    Reduction
        identity [0.000000f]
        lhs [x]  rhs [y]
        combiner [(x + y)]
        axes (k : [0, 9999])
        condition (uint1)1
        source[0] = (X(i, k)*W(j, k))
    
    tensor X{0x23c4660}[0] : float32 [32, 10000]
        placeholder(X, 0x23c4660)
    
    tensor W{0x22a07e0}[0] : float32 [3000, 10000]
        placeholder(W, 0x22a07e0)
    
    
    
    Jacobian(Y, W):
    tensor compute.jacobian{0x2386970}[0] : float32 [32, 3000, 3000, 10000]
    axes (i : [0, 31], j : [0, 2999], jac_i0 : [0, 2999], jac_i1 : [0, 9999])
    Reduction
        identity [0.000000f]
        lhs [x.der]  rhs [y.der]
        combiner [(x.der + y.der)]
        axes (k : [0, 9999])
        condition (uint1)1
        source[0] = (X(i, k)*float32(((jac_i0 == j) && (jac_i1 == k))))
    
    tensor X{0x23c4660}[0] : float32 [32, 10000]
        placeholder(X, 0x23c4660)
    
    


You can see that `W(j, k)` in the original tensor Y became `float32(((jac_i0 == j) && (jac_i1 == k)))` in the Jacobian, which is the derivative of `W(j, k)` wrt `W(jac_i0, jac_i1)` (it's equal to 1 if the corresponding indices coincide, otherwise it's zero). Of course, computing this Jacobian is very inefficient, because it consists of summing over mostly zero values, so it should be optimized by propagating the information that `jac_i1 == k` and completely removing the summation. It may be done with the function `OptimizeAndLiftNonzeronessConditions` (which is called by the `Jacobian` function by default). Let's call it manually:


```python
dYdW_optimized = tvm.ir_pass.OptimizeAndLiftNonzeronessConditions(dYdW)
print(tvm.PrintTensorRecursively(dYdW_optimized))
```

    tensor compute.jacobian{0x2363500}[0] : float32 [32, 3000, 3000, 10000]
    axes (i : [0, 31], j : [0, 2999], jac_i0 : [0, 2999], jac_i1 : [0, 9999])
        select(((((((jac_i1 <= 9999) && (i <= 31)) && (j == jac_i0)) && (j <= 2999)) && (jac_i0 <= 2999)) && (jac_i1 <= 9999)), (X(i, jac_i1)*1.000000f), 0.000000f)
    
    tensor X{0x23c4660}[0] : float32 [32, 10000]
        placeholder(X, 0x23c4660)
    
    


The reduction was eliminated completely, and replaced with a conditional expression returning `X(i, jac_i1)` if `j == jac_i0` and 0 otherwise. You can see a small deficiency here: the condition contains redundant formulas which are implied by variable boundaries (they are usually eliminated in subsequent passes though).

The condition `j == jac_i0` may be used to eliminate another reduction. Recall that the Jacobian is used in a formula looking similar to `matmul(H, Jacobian(Y, W))`, so the reduction to be eliminated is a summation used in matrix multiplication. To perform this transformation, `JacobianRecursive` inlines the Jacobian and calls `OptimizeAndLiftNonzeronessConditions` once more. Let's do this manually:


```python
# Generalized matmul works with tensors of arbitrary dimensions and takes
# an additional parameter: the number of dimensions to contract. It is
# semantically equivalent to reshaping into two matrices, 
# performing matrix multiplication, and then reshaping back
dLdW = tvm.generalized_matmul(H, dYdW_optimized, 2)

# We have to inline dYdW_optimized because OptimizeAndLiftNonzeronessConditions works
# only with a single tensor
dLdW_inlined = tvm.ir_pass.InlineNonReductions(dLdW, [dYdW_optimized])

# Perform the main optimization
dLdW_optimized = tvm.ir_pass.OptimizeAndLiftNonzeronessConditions(dLdW_inlined)
print(tvm.PrintTensorRecursively(dLdW_optimized))
```

    tensor tensor{0x23ef840}[0] : float32 [3000, 10000]
    axes (ax0 : [0, 2999], ax1 : [0, 9999])
        select(((ax0 <= 2999) && (ax1 <= 9999)), auto.extracted_reduction(ax0, ax1), 0.000000f)
    
    tensor auto.extracted_reduction{0x248b4d0}[0] : float32 [3000, 10000]
    axes (ax0 : [0, 2999], ax1 : [0, 9999])
    Reduction
        identity [0.000000f]
        lhs [x]  rhs [y]
        combiner [(x + y)]
        axes (k0.shifted : [0, 31])
        condition (k0.shifted <= 31)
        source[0] = (H(k0.shifted, ax0)*(X(k0.shifted, ax1)*1.000000f))
    
    tensor H{0x2212a80}[0] : float32 [32, 3000]
        placeholder(H, 0x2212a80)
    
    tensor X{0x23c4660}[0] : float32 [32, 10000]
        placeholder(X, 0x23c4660)
    
    


You can see that now there is only one reduction axis left, an there no comparison `j == jac_i0` anymore. You can also see another quirk: `OptimizeAndLiftNonzeronessConditions` has split our tensor into two parts: one just contains a condition (quite useless), and another is a reduction. The reason is that this function (as you can guess from its name) lifts nonzeroness conditions up, and sometimes it factors them out of a reduction, and because of some tvm restrictions the reduction must be extracted into a separate tensor. (Note that in the case of summation the condition could be put into the `condition` field of the reduction, but this wouldn't work for different reductions).

# Supported operations
Here is a list of operations which seem to be differentiated quite well by our autodiff.

## Dense


```python
X = tvm.placeholder((32, 1000), name='X')
W = tvm.placeholder((1000, 1000), name='W')
B = tvm.placeholder((1000,), name='B')

Y = topi.nn.dense(X, W, B)

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.ir_pass.JacobianRecursive(Y, [X, W, B], H)
```


```python
print("forward ", measure_performance([Y], [X, B, W]))
print("backward", measure_performance(list(grads), [X, B, W, H]))
```

    forward  28
    backward 60


## Conv2D


```python
X = tvm.placeholder((32, 17, 28, 28), name='X')
W = tvm.placeholder((19, 17, 3, 3), name='W')
Y = topi.nn.conv2d(X, W, [1, 1], [0, 0])

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.ir_pass.JacobianRecursive(Y, [X, W], H)
```


```python
print("forward ", measure_performance([Y], [X, W]))
print("backward", measure_performance(list(grads), [X, W, H]))
```

    forward  52
    backward 118


# Somewhat supported

## Average pooling
The performance is suspicious but the generated code looks ok except for large if expressions which cannot be eliminated by subsequent passes (the problem in not nearly as horrible as with max pooling).


```python
X = tvm.placeholder((32, 17, 280, 280), name='X')
Y = topi.nn.pool(X, [2, 2], [2, 2], [0, 0, 0, 0], 'avg')

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.ir_pass.JacobianRecursive(Y, [X], H)
```


```python
print("forward ", measure_performance([Y], [X]))
print("backward", measure_performance(list(grads), [X, H]))
```

    forward  14
    backward 233



```python
print(tvm.PrintTensorRecursively(grads[0]))
```

    tensor tensor.grad{0x2d73500}[0] : float32 [32, 17, 280, 280]
    axes (ax0 : [0, 31], ax1 : [0, 16], ax2 : [0, 279], ax3 : [0, 279])
        select(((((((((((((((((ax0 <= 31) && (ax1 <= 16)) && (ax2 <= 279)) && (ax3 <= 279)) && ((ax2 - (((ax2 + -1)/2)*2)) <= 3)) && (-1 <= ax2)) && (((((ax2 + -1)/2)*2) - ax2) <= 0)) && (0 <= ax2)) && (ax2 <= 280)) && (ax2 <= 279)) && ((ax3 - (((ax3 + -1)/2)*2)) <= 3)) && (-1 <= ax3)) && (((((ax3 + -1)/2)*2) - ax3) <= 0)) && (0 <= ax3)) && (ax3 <= 280)) && (ax3 <= 279)), auto.extracted_reduction(ax0, ax1, ax2, ax3), 0.000000f)
    
    tensor auto.extracted_reduction{0x27e8dd0}[0] : float32 [32, 17, 280, 280]
    axes (ax0 : [0, 31], ax1 : [0, 16], ax2 : [0, 279], ax3 : [0, 279])
    Reduction
        identity [0.000000f]
        lhs [x]  rhs [y]
        combiner [(x + y)]
        axes (k3.shifted : [0, 1], k2.shifted : [0, 1])
        condition (((((((((ax2 - 1) <= ((k2.shifted*2) + (((ax2 - 1)/2)*2))) && ((0 - ((ax2 - 1)/2)) <= k2.shifted)) && (((k2.shifted*2) + (((ax2 - 1)/2)*2)) <= ax2)) && ((k2.shifted + ((ax2 - 1)/2)) <= 139)) && ((ax3 - 1) <= ((k3.shifted*2) + (((ax3 - 1)/2)*2)))) && ((0 - ((ax3 - 1)/2)) <= k3.shifted)) && (((k3.shifted*2) + (((ax3 - 1)/2)*2)) <= ax3)) && ((k3.shifted + ((ax3 - 1)/2)) <= 139))
        source[0] = (H(ax0, ax1, (k2.shifted + ((ax2 - 1)/2)), (k3.shifted + ((ax3 - 1)/2)))*(1.000000f*0.250000f))
    
    tensor H{0x2480340}[0] : float32 [32, 17, 140, 140]
        placeholder(H, 0x2480340)
    
    


## Softmax (homebrewn)
Softmax from topi causes performance problems (see below), but we can write our own softmax which works better but still not perfectly (seems like some performance problems when used after other layers like dense).


```python
X = tvm.placeholder((60, 100), name="X")
W = tvm.placeholder((1000, 1000), name='W')

exps = topi.exp(topi.nn.dense(X, W))
sumexps = topi.sum(exps, axis=-1, keepdims=True)
Y = exps/sumexps

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.ir_pass.JacobianRecursive(Y, [X, W], H)
```


```python
print("forward ", measure_performance([Y], [X, W]))
print("backward", measure_performance(list(grads), [X, W, H]))
```

    forward  10
    backward 223


# Completely unsupported

## Flatten
Flatten uses the division and modulo operations which are not well supported by our zero-eliminating transformations. A related problem is [issue 1711](https://github.com/dmlc/tvm/issues/1711).


```python
X = tvm.placeholder((32, 10, 20, 25), name='X')
Y = topi.nn.flatten(X)

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.ir_pass.JacobianRecursive(Y, [X], H)
```


```python
print("forward ", measure_performance([Y], [X]))
print("backward", measure_performance(list(grads), [X, H]))
```

    forward  0
    backward 1568



```python
print(tvm.PrintTensorRecursively(grads[0]))
```

    tensor compute.grad{0x26ae2a0}[0] : float32 [32, 10, 20, 25]
    axes (ax0 : [0, 31], ax1 : [0, 9], ax2 : [0, 19], ax3 : [0, 24])
        select(((((ax0 <= 31) && (ax1 <= 9)) && (ax2 <= 19)) && (ax3 <= 24)), auto.extracted_reduction(ax0, ax1, ax2, ax3), 0.000000f)
    
    tensor auto.extracted_reduction{0x2716d20}[0] : float32 [32, 10, 20, 25]
    axes (ax0 : [0, 31], ax1 : [0, 9], ax2 : [0, 19], ax3 : [0, 24])
    Reduction
        identity [0.000000f]
        lhs [x]  rhs [y]
        combiner [(x + y)]
        axes (k1.shifted : [0, 4999])
        condition ((((k1.shifted <= 4999) && (ax1 == ((k1.shifted/500) % 10))) && (ax2 == ((k1.shifted/25) % 20))) && (ax3 == (k1.shifted % 25)))
        source[0] = (H(ax0, k1.shifted)*1.000000f)
    
    tensor H{0x2d73890}[0] : float32 [32, 5000]
        placeholder(H, 0x2d73890)
    
    


Here the compiler has to figure out that `k1.shifted` is directly expressible using `ax1, ax2, ax3`.

## Max pooling
Reducing with other combiners, like max, is a bit trickier than summation. Currently reducing with max is partially supported: it can be differentiated, and most of transformations, like moving some conditions out of the resulting reduction, work, but still there are some transformations left to be implemented to make it performant. In particular, autodifferentiated max pooling cannot even be compiled in most practical cases, because it involves creating tensors larger than 2^31.


```python
X = tvm.placeholder((1, 2, 100, 100), name='X')
Y = topi.nn.pool(X, [2, 2], [2, 2], [0, 0, 0, 0], 'max')

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.ir_pass.JacobianRecursive(Y, [X], H)
```


```python
print("forward ", measure_performance([Y], [X]))
print("backward", measure_performance(list(grads), [X, H]))
```

    forward  0
    backward 475



```python
print(tvm.PrintTensorRecursively(grads[0]))
```

    tensor tensor.grad{0x24ae0b0}[0] : float32 [1, 2, 100, 100]
    axes (ax0 : [0, 0], ax1 : [0, 1], ax2 : [0, 99], ax3 : [0, 99])
        select(((((((((((((((((ax0 == 0) && (ax1 <= 1)) && (ax2 <= 99)) && (ax3 <= 99)) && ((ax2 - (((ax2 + -1)/2)*2)) <= 3)) && (-1 <= ax2)) && (((((ax2 + -1)/2)*2) - ax2) <= 0)) && (0 <= ax2)) && (ax2 <= 100)) && (ax2 <= 99)) && ((ax3 - (((ax3 + -1)/2)*2)) <= 3)) && (-1 <= ax3)) && (((((ax3 + -1)/2)*2) - ax3) <= 0)) && (0 <= ax3)) && (ax3 <= 100)) && (ax3 <= 99)), auto.extracted_reduction(ax0, ax1, ax2, ax3), 0.000000f)
    
    tensor auto.extracted_reduction{0x2481f30}[0] : float32 [1, 2, 100, 100]
    axes (ax0 : [0, 0], ax1 : [0, 1], ax2 : [0, 99], ax3 : [0, 99])
    Reduction
        identity [0.000000f]
        lhs [x]  rhs [y]
        combiner [(x + y)]
        axes (k3.shifted : [0, 1], k2.shifted : [0, 1])
        condition (((((((((ax2 - 1) <= ((k2.shifted*2) + (((ax2 - 1)/2)*2))) && ((0 - ((ax2 - 1)/2)) <= k2.shifted)) && (((k2.shifted*2) + (((ax2 - 1)/2)*2)) <= ax2)) && ((k2.shifted + ((ax2 - 1)/2)) <= 49)) && ((ax3 - 1) <= ((k3.shifted*2) + (((ax3 - 1)/2)*2)))) && ((0 - ((ax3 - 1)/2)) <= k3.shifted)) && (((k3.shifted*2) + (((ax3 - 1)/2)*2)) <= ax3)) && ((k3.shifted + ((ax3 - 1)/2)) <= 49))
        source[0] = (H(0, ax1, (k2.shifted + ((ax2 - 1)/2)), (k3.shifted + ((ax3 - 1)/2)))*auto.extracted_reduction(0, ax1, (k2.shifted + ((ax2 - 1)/2)), (k3.shifted + ((ax3 - 1)/2)), ax0, ax1, ax2, ax3))
    
    tensor H{0x27df8c0}[0] : float32 [1, 2, 50, 50]
        placeholder(H, 0x27df8c0)
    
    tensor auto.extracted_reduction{0x249c710}[0] : float32 [1, 2, 50, 50, 1, 2, 100, 100]
    axes (ax0 : [0, 0], ax1 : [0, 1], ax2 : [0, 49], ax3 : [0, 49], jac_i0 : [0, 0], jac_i1 : [0, 1], jac_i2 : [0, 99], jac_i3 : [0, 99])
    Reduction
        identity [0.000000f, -340282346638528859811704183484516925440.000000f]
        lhs [x.der, x]  rhs [y.der, y]
        combiner [((x.der*select((x < y), 0.000000f, 1.000000f)) + (y.der*select((x < y), 1.000000f, 0.000000f))), max(x, y)]
        axes (rv.shifted : [0, 1], rv.shifted : [0, 1])
        condition ((rv.shifted <= 1) && (rv.shifted <= 1))
        source[0] = select(((((ax3*2) + rv.shifted) == jac_i3) && ((rv.shifted + (ax2*2)) == jac_i2)), 1.000000f, 0.000000f)
        source[1] = X(ax0, ax1, ((ax2*2) + rv.shifted), ((ax3*2) + rv.shifted))
    
    tensor X{0x23b9c20}[0] : float32 [1, 2, 100, 100]
        placeholder(X, 0x23b9c20)
    
    


Looking at the last tensor we can see that it must be zero for most values of `jac_i2` and `jac_i3`. They can actually be replaced with `jac_i2 - ax2*2` and `jac_i3 - ax3*2` which can be proved to have much smaller range. A similar transformation is successfully performed in the average pooling case, but here the situation is slightly different causing it to fail.

## Softmax
Softmax uses max behind the scenes, causing some performance problems. We havent't yet investingated into it though.


```python
X = tvm.placeholder((60, 200), name="X")
Y = topi.nn.softmax(X)

H = tvm.placeholder(Y.shape, name='H')
grads = tvm.ir_pass.JacobianRecursive(Y, [X], H)
```


```python
print("forward ", measure_performance([Y], [X]))
print("backward", measure_performance(list(grads), [X, H]))
```

    forward  0
    backward 785

