
# Training some neural net on mnist with pure tvm

Here we are going to train a neural net in pure tvm (no nnvm) using tvm-level automatic differentiation. We also implement the same network in keras and use it as a reference implementation. Some operations, like max pooling and flatten are not fully supported yet, so we avoid using them. Also note that we do little scheduling here, and since we don't use nnvm, some optimization are not performed, so it's all going to be quite slow. In real life, tvm-level autodiff is supposed to be used on individual nnvm/relay operations together with some kind of automatic scheduling, may be autotuning.


```python
import topi
import tvm
import matplotlib.pyplot as plt

import numpy as np
import itertools
import time
```

Load data using keras utility functions.


```python
batch_size = 32
num_classes = 10

import os

import tensorflow as tf
import keras
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
```

    /nix/store/p3i9s3vhjskbrnfl97fd7b0vmn7bqddh-python3.6-h5py-2.7.1/lib/python3.6/site-packages/h5py/__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.


Batch generator. The last incomplete batch is thrown out because we use fixed batch size. We will use the same function for keras so that the training results are closer.


```python
def batches(x, y):
    for i in range(int(x.shape[0] / batch_size)):
        yield (x[i:i+batch_size, :, :, None].astype('float32'),
               y[i:i+batch_size, ...].astype('float32'))
```

## Defining the model

This is the keras definition of the model.


```python
def make_keras_model():
    data = keras.layers.Input(shape=(28, 28, 1))
    x = data
    x = keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(x)
    x = keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dense(num_classes, activation='softmax')(x)
    keras_model = keras.models.Model(data, x)

    keras_model.compile(loss=keras.losses.categorical_crossentropy, metrics=['accuracy'],
                        optimizer=keras.optimizers.SGD(lr=1e-2))
    
    return keras_model

keras_model = make_keras_model()
```


```python
keras_model.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         (None, 28, 28, 1)         0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 9216)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               1179776   
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    _________________________________________________________________


This is the same thing written in tvm. Note that we use a custom implementation of softmax because the topi implementation is too complex for our automatic differentiation.


```python
weights = []

x = tvm.placeholder((batch_size, 28, 28, 1))
y = tvm.placeholder((batch_size, num_classes))

t = topi.transpose(x, [0, 3, 1, 2])

w1 = tvm.placeholder((32, 1, 3, 3), name="w1")
b1 = tvm.placeholder((32,), name="b1")
t = topi.nn.relu(topi.nn.conv2d(t, w1, 1, 0, 1) + topi.reshape(b1, (1, 32, 1, 1)))
weights.extend([w1, b1])

w2 = tvm.placeholder((64, 32, 3, 3), name="w2")
b2 = tvm.placeholder((64,), name="b2")
t = topi.nn.relu(topi.nn.conv2d(t, w2, 1, 0, 1) + topi.reshape(b2, (1, 64, 1, 1)))
weights.extend([w2, b2])

t = topi.nn.pool(t, [2, 2], [2, 2], [0, 0, 0, 0], 'max')

# Note that we have to transpose before flatten
t = topi.transpose(t, [0, 2, 3, 1])
t = topi.nn.flatten(t)

w3 = tvm.placeholder((128, np.prod([s.value for s in t.shape[1:]])), name="w3")
b3 = tvm.placeholder((128,), name="b3")
t = topi.nn.relu(topi.nn.dense(t, w3, b3))
weights.extend([w3, b3])

w4 = tvm.placeholder((num_classes, 128), name="w4")
b4 = tvm.placeholder((num_classes,), name="b4")
t = topi.nn.dense(t, w4, b4)
weights.extend([w4, b4])

# We use a custom softmax because the topi implementation uses max behind the scenes
# which currently causes problems with autodiff, leading to poor performance
exps = topi.exp(t)
sumexps = topi.sum(exps, axis=-1, keepdims=True)
logsoftmax = topi.log(exps/sumexps)

predictions = topi.nn.softmax(t)
loss = - topi.sum(y * logsoftmax) / batch_size
```

Here we perform automatic differentiation of the loss wrt the weights.


```python
# We'll get gradients because loss is a scalar.
gradients = list(tvm.differentiate(loss, weights))

# We feed the learning rate as an input.
learning_rate = tvm.placeholder(())

# For simplicity we compute new weights as an output and then copy the result to the input
new_weights = [w - learning_rate*g for w, g in zip(weights, gradients)]
```

## Compiling and initializing the weights

A couple of helper functions


```python
# get tensor's shape as a list
def get_shape(tensor):
    return [s.value for s in tensor.shape]

# empty tensor values for a list of tensors or just a single tensor
def empty_val(tensor):
    if isinstance(tensor, list):
        return [empty_val(t) for t in tensor]
    else:
        return tvm.nd.empty(get_shape(tensor), tensor.dtype)
```

Just assume that we have 20 cores and parallelize it somehow. (TODO: find something better)


```python
def schedule_somehow(sched):
    # This autoinlining turned aoyt to be harmful because it prevents memoization of some expensive operations
    # tvm.schedule.AutoInlineInjective(sched)
    for s in sched.stages:
        if isinstance(s.op, tvm.tensor.ComputeOp) and isinstance(s.op.body[0], tvm.expr.Reduce):
            ax = s.fuse(*s.op.axis)
            axo, axi = s.split(ax, nparts=20)
            s.parallel(axo)
```

Build two separate modules: one for testing and one for training.


```python
sched = tvm.create_schedule(loss.op)
schedule_somehow(sched)
testing_module = tvm.build(sched, [loss, x, y] + weights)
```


```python
sched = tvm.create_schedule([loss.op] + [w.op for w in new_weights])
schedule_somehow(sched)
training_module = tvm.build(sched, [loss, x, y, learning_rate] + new_weights + weights)
```

This class just stores the current state of weights and provides a couple of useful methods.


```python
class TvmModel:
    def __init__(self):
        self.weights_values = empty_val(weights)
        
    def test(self, xval, yval):
        args = [empty_val(loss)] + [tvm.ndarray.array(xval), tvm.ndarray.array(yval)] + self.weights_values
        testing_module(*args)
        return args[0].asnumpy()

    def train(self, xval, yval, lr=1e-2):
        new_weights_values = empty_val(new_weights)
        args = [empty_val(loss)] + [tvm.ndarray.array(xval.astype('float32')), tvm.ndarray.array(yval.astype('float32'))] +\
                [tvm.ndarray.array(np.array(lr).astype('float32'))] + new_weights_values + self.weights_values
        training_module(*args)
        for wv, new_wv in zip(self.weights_values, new_weights_values):
            wv.copyfrom(new_wv)
        return args[0].asnumpy()
    
    def from_keras(self, keras_model):
        assert len(keras_model.get_weights()) == len(self.weights_values)
        for kv, wv in zip(keras_model.get_weights(), self.weights_values):
            if len(kv.shape) == 4:
                kv = np.transpose(kv, [3, 2, 0, 1])
            elif len(kv.shape) == 2:
                kv = np.transpose(kv)

            #print(wv.shape, " <- ", kv.shape)

            wv.copyfrom(kv)
            
    def to_keras(self, keras_model):
        assert len(keras_model.get_weights()) == len(self.weights_values)
        new_keras_weights = []
        for kv, wv in zip(keras_model.get_weights(), self.weights_values):
            wv_np = wv.asnumpy()
            if len(kv.shape) == 4:
                wv_np = np.transpose(wv_np, [2, 3, 1, 0])
            elif len(kv.shape) == 2:
                wv_np = np.transpose(wv_np)
            new_keras_weights.append(wv_np)

        keras_model.set_weights(new_keras_weights)
```

Create a tvm model and copy weights from the keras model as initialization. Check that transferring weights between tvm and keras works.


```python
tvm_model = TvmModel()
keras_model_to_check = make_keras_model()

tvm_model.from_keras(keras_model)
tvm_model.to_keras(keras_model_to_check)

for w2, w1 in zip(keras_model_to_check.get_weights(), keras_model.get_weights()):
    np.testing.assert_allclose(w2, w1)
```

Testing on batches should produce the same result


```python
for xx, yy in itertools.islice(batches(x_train, y_train), 10):
    print(tvm_model.test(xx, yy), "vs", keras_model_to_check.test_on_batch(xx, yy))
```

    2.291964 vs [2.2919638, 0.125]
    2.29213 vs [2.2921298, 0.125]
    2.2945328 vs [2.2945325, 0.125]
    2.2937841 vs [2.2937841, 0.125]
    2.2963736 vs [2.2963734, 0.125]
    2.2951095 vs [2.2951093, 0.125]
    2.2955298 vs [2.2955298, 0.125]
    2.2995322 vs [2.2995322, 0.125]
    2.3017006 vs [2.3017006, 0.125]
    2.3027904 vs [2.3027906, 0.125]


Let's compare test (forward pass) time on several batches to know what to expect (on my machine keras is about 2x faster).


```python
%%time
for xx, yy in itertools.islice(batches(x_train, y_train), 1000):
    keras_model_to_check.test_on_batch(xx, yy)
```

    CPU times: user 1min 49s, sys: 2min 46s, total: 4min 36s
    Wall time: 13.3 s



```python
%%time
for xx, yy in itertools.islice(batches(x_train, y_train), 1000):
    tvm_model.test(xx, yy)
```

    CPU times: user 7min 14s, sys: 50.5 s, total: 8min 5s
    Wall time: 24.3 s


## Training the reference keras model

Let's first train the reference keras model. We use our custom batch generator to make the comparison fairer.


```python
keras_model.fit_generator(batches(x_train, y_train), steps_per_epoch=int(len(x_train) / batch_size))
```

    Epoch 1/1
    1875/1875 [==============================] - 78s 42ms/step - loss: 0.1902 - acc: 0.9619





    <keras.callbacks.History at 0x7f2408427b38>



## Training the tvm model
Train the tvm model. Note that train loss here is the loss from the last step, so don't compare it to the keras train loss. On my machine training is about 5x slower than training with keras instead of being just 2x slower as with testing. 


```python
start_time = time.time()
seen = 0
for step, (xs, ys) in enumerate(batches(x_train, y_train)):
    seen += xs.shape[0]
    train_loss = tvm_model.train(xs, ys)
    
    cur_time = time.time()
    
    ms_per_step = int(1000*(cur_time - start_time)/(step + 1))
    
    print("samples: {}  step: {}  {}ms/step  train loss: {}".format(seen, step, ms_per_step, train_loss), end='\r')
        
print("")
```

    samples: 60000  step: 1874  107ms/step  train loss: 0.0126412529498338743


## Testing
We'll test both models using keras, the results are lists `[loss, accuracy]`.


```python
keras_model.evaluate_generator(batches(x_test, y_test), steps=int(len(x_test) / batch_size))
```




    [0.3763718899005117, 0.8784054487179487]




```python
tvm_model.to_keras(keras_model_to_check)
keras_model_to_check.evaluate_generator(batches(x_test, y_test), steps=int(len(x_test) / batch_size))
```




    [0.36809223694488025, 0.8738982371794872]


