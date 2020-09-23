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


## Loss class

A loss class is an implementation of interface `minet.loss.Loss`. Similar to 
a layer class, we have to implement forward and backward
(note that the output of forward is a scalar rather than a matrix, 
and backward doesn't need to take `dl/dY` as input)
```java
public double forward(DoubleMatrix Y, DoubleMatrix Yhat);
public DoubleMatrix backward();
```


The difference is that a loss class doesn't contain any weights, thus 
we don't need to collect its weights and gradients.


## Excercise
1. Implement a layer class for [tanh activation function](https://en.wikipedia.org/wiki/Activation_function).

2. Implement a loss class for [Hinge loss](https://en.wikipedia.org/wiki/Hinge_loss).  
