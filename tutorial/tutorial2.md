# Tutorial 2: Adding a new layer and a loss function

The purpose of this tutorial is to get used to implementing layers and 
loss functions using matrix notations and [jblas](http://jblas.org/)
functions.

*Note that: following [pytorch](pytorch.org) and several other deep learning 
libraries, an instance is represented by a *row* vector. 
Therefore, a linear layer is Y = X * W + b rather than Y = W * X + b.*

## Layer class

A layer class is an implementation of interface `minet.layer.Layer`
with four following functions

```java
// Computing the output of applying the layer to input X, i.e. Y = Layer(X).
public DoubleMatrix forward(DoubleMatrix X);

// Computing the gradient of the layer's parameters and the input when applying forward.
public DoubleMatrix backward(DoubleMatrix gY);

// Collect all the weight matrices and bias vectors of the layer.
public List<DoubleMatrix> getAllWeights(List<DoubleMatrix> weights);

// Collect the gradients of all the weight matrices and bias vectors of the layer.
public List<DoubleMatrix> getAllGradients(List<DoubleMatrix> grads);
``` 

Let's look at the implement of class `minet.layer.Linear`
(see [here](http://cs231n.stanford.edu/handouts/linear-backprop.pdf)
for mathematical details).

### Forward
A linear layer contains a weight matrix `W` and a bias row vector `b`.
Given a matrix X whose each row is an instance x. The forward is 
to compute
```
Y = X * W + b
```
It is implemented as follows
```java
public DoubleMatrix forward(DoubleMatrix X) {
    DoubleMatrix Y = X.mmul(this.W).addiRowVector(this.b);
    this.X = X.dup();
    return Y;
}
```
Here, firstly we perform  `X * W` (matrix multiplication) by `X.mmul(this.W)`. 
We then add `b` to every row of the resulting matrix by `addiRowVector(this.b)`.

To confirm the correctness of the function, we can compare its output 
against the implementation of some well-known deep learning library e.g. PyTorch, 
Tensorflow. Or, we can use math software e.g. Matlab.

### Backward
Now, assuming that we have computed a loss `l` as a function of `Y`, and 
`dl/dY` (each row is gradient row vector `dl/dy`). 
The backward is to compute  
```
dl/dW = X^T * dl/dY
dl/db = sum_row dl/dY
dl/dX = dl/dY * W^T
```
Turning that to java language, we have
```java
public DoubleMatrix backward(DoubleMatrix gY) {
    this.gW.addi(this.X.transpose().mmul(gY));
    this.gb.addi(gY.columnSums());
    return gY.mmul(this.W.transpose());
}
```
Note that `A.addi(B)` is to add `B` directly to `A`. It means that in the code, `this.gW` and 
`this.gb` are updated with the new gradients. This is because one might want to keep the gradients
which have been computed before. In our case, as we don't need that, we have to reset the gradients 
(by preforming `optimizer.resetGradients()`) before calling `backward`.

A common way to make sure that the backward function is correct is to perform 
[gradient checking](http://deeplearning.stanford.edu/tutorial/supervised/DebuggingGradientChecking/), 
which is to compare the computed gradients against their numerical approximate. 
*Note that this kind of checking requires high precision; hence, `double` 
is used.*

Performing gradient checking is easy thanks to `minet.GradientChecker`. 
Two examples are given in `minet.GradientChecker.Test1` and `Test2`.
```java
public static void test1() {
    // some instances (input X and output Y)
    DoubleMatrix X = new DoubleMatrix(
            new double[][] {
                    {.1f, .1f, .1f, .6f, .1f},
                    {.5f, .1f, .2f, .1f, .1f},
                    {.1f, .2f, .2f, .1f, .4f}});
    DoubleMatrix Y = new DoubleMatrix(new double[][] {
            {2., 0.},
            {-0.1, 5.},
            {3., -1.2}});

    // create a network that contains only the layer needed to be checked
    Sequential net = new Sequential(new Layer[]{
            new Linear(5, 2, new Linear.WeightInitUniform(-1, 1))
    });

    // we use MSE loss as it is confirmed to be correct
    MeanSquaredError loss = new MeanSquaredError();

    System.out.println(net);
    System.out.println(loss);

    // gradient check
    checkGradient(net, loss, X, Y);
}
```
If we see the output
```
(
    Linear: 5 in, 2 out
)
MeanSquareErrorLoss
correct backward for input
correct backward for weights
```
our backward implementation passed the gradient checking test. 

## Loss class

A loss class is an implementation of interface `minet.loss.Loss`. Similar to 
a layer class, we have to implement forward and backward
```java
public double forward(DoubleMatrix Y, DoubleMatrix Yhat);
public DoubleMatrix backward();
```
The differences are
- the output of forward is a scalar rather than a matrix, 
and backward doesn't need to take `dl/dY` as input, 
- a loss class doesn't contain any weights, thus 
we don't need to collect weights and gradients.


## Excercise
1. Implement a layer class for [tanh activation function](https://en.wikipedia.org/wiki/Activation_function).

2. Implement a loss class for [Hinge loss](https://en.wikipedia.org/wiki/Hinge_loss).  

*(Don't forget to use gradient checking!)*
