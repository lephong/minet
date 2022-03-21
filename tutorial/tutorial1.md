# Tutorial 1: Building an Artificial Neural Network (ANN)

In this tutorial, we will walk the reader step by step how to build (train and test) 
a handwritten digit recognizer using [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
The source code is available in `minet/example/mnist/`. 
  
## Preparing data
We already prepared the dataset, which can be found in `minet/example/mnist/data.zip` 
There are three files 
```
train.txt
dev.txt
test.txt
```
Each file starts with a line telling us `n` - the number of instances in that file,
`dx` - the dimensionality of images. Each subsequent line contains `dx` float numbers representing an image, 
followed by the ground-truth digit of that image. For instance, the training file is

```
50000 784
0.0 ... 0.9921568627450981 0.9921568627450981 ... 0.0 ; 5
...
0.0 ... 0.24313725490196078 0.3176470588235294 ... 0.0 ; 4
...
```
meaning that it contains 50000 instances, each image is represented by 784 floats. 
The first image is digit 5.

In general, the format is 

```
[n = Number of instances] [dx = dims of x]
[dx floats, separated by " "] ; [dy floats, separated by " "]
...
[dx floats, separated by " "] ; [dy floats, separated by " "]
...
```

## Loading the data: create an implementation of `minet.data.Dataset`
We use `minet.data.Dataset` to store datasets and 
iterator over mini-batches. 

We will need to write an `MNISTDataset` class that implements `minet.data.Dataset`. 
The aim of the class is to support loading an MNIST dataset from a given data file. 
We will implement two methods in this class:

- `int getInputDims()`: get the number of input features
- `fromFile(String path)`: load an MNIST dataset from file `path` into `items` (see `minet.data.Dataset`)


## Building an ANN

We will create a new class called `MNISTClassifier`. All steps for building and training an ANN model for MNIST will be located in this class.

We build a one-hidden-layer neural network classifier: 
- The input layer has 784 nodes, corresponding to 784 dims of x. 
- The hidden layer has 1000 nodes, with ReLU activation function.
- The output layer has 10 nodes, corresponding to 10 digits. 
Because the task is classification, 
we use a softmax to compute a distribution over 
the 10 digits. 

```java
int indims = trainset.getInputDims();  // get the dims of x (784)
int hiddims = 1000; 
int nClasses = 10;  // there are 10 digits (i.e. 10 output classes)
Sequential net = new Sequential(new Layer[] {
        new Linear(indims, hiddims, new Linear.WeightInitXavier()),
        new ReLU(),
        new Linear(hiddims, nClasses, new Linear.WeightInitXavier()),
        new Softmax()});
```
We use cross-entropy loss to train the network

```java
CrossEntropy loss = new CrossEntropy();
```
    
and stochastic gradient descent to minimize the loss function

```java
double learningRate = 0.1;
Optimizer sgd = new SGD(net, learningRate);
```

Finally, for logging, 
we print out the structure of the network and the loss 

```java
System.out.println(net);
System.out.println(loss);
```

## Convert each mini-batch to structures supported by our network

Each MNIST mini-batch returned by `minet.data.Dataset.getNextMiniBatch()` is a list of items, each item consists of an input and its true label. Because the batch is given as input to the first hidden layer of our ANN, and that layer (`minet.layer.Linear`) expect a batch input as a `DoubleMatrix` object, we will convert this list to two DoubleMatrix objects, the first one is an input matrix (#rows = #items, #cols = #input features) and an output matrix (#rows = #items, #cols = 1).


```java
Pair<DoubleMatrix, DoubleMatrix> fromBatch(List<Pair<double[], Integer>> batch) {
    if (batch == null)
        return null;
    
    double[][] xs = new double[batch.size()][];
    double[] ys = new double[batch.size()];
    for (int i = 0; i < batch.size(); i++) {
        xs[i] = batch.get(i).first; 
        ys[i] = (double)batch.get(i).second;
    }
    DoubleMatrix X = new DoubleMatrix(xs);
    DoubleMatrix Y = new DoubleMatrix(ys.length, 1, ys);
    return new Pair<DoubleMatrix, DoubleMatrix>(X, Y);
}
```

    
## Evaluating the network
For classification, we evaluate the network using accuracy metric
`accuracy = #correct / #total`. Other widely used metrics include 
[precision, recall, and F1](https://en.wikipedia.org/wiki/Precision_and_recall). 

```java
double eval(Layer net, MNISTDataset data) {
    // reset index of the data
    data.reset();
    
    // the number of correct predictions so far
    double correct = 0;

    while (true) {
        // we evaluate per mini-batch by making use of the fromBatch method implemented above
        Pair<DoubleMatrix, DoubleMatrix> batch = fromBatch(data.getNextMiniBatch());
        if (batch == null)
            break;

        // perform forward pass to compute Yhat (the predictions)
        // each row of Yhat is a probabilty distribution over 10 digits
        DoubleMatrix Yhat = net.forward(batch.first);

        // the predicted digit for each image is the one with the highest probability
        int[] preds = Yhat.rowArgmaxs();

        // count how many predictions are correct
        for (int i = 0; i < preds.length; i++) {
            if (preds[i] == (int) batch.second.data[i])
                correct++;
        }
    }

    // compute classification accuracy
    double acc = correct / data.getSize();
    return acc;
}
```


## Training the network
We train the network by updating its weights after processing 
each mini-batch. After each epoch, we evaluate the current weights
on a development set. If the accuracy is not at peak `patience` times 
consecutively, we halt the training (this is an 
[early stopping](https://en.wikipedia.org/wiki/Early_stopping)
strategy).

Note:
- always *shuffle* the training data before each epoch (already done inside `minet.data.Dataset`)
- always *reset* the gradients before performing backward.

```java
void train(Layer net, Loss loss, Optimizer optimizer, MNISTDataset traindata,
                         MNISTDataset devdata, int nEpochs, int patience) {
    int notAtPeak = 0;  // the number of times not at peak
    double peakAcc = -1;  // the best accuracy of the previous epochs
    double totalLoss = 0;  // the total loss of the current epoch

    traindata.reset(); // reset index and shuffle the data before training
    
    for (int e = 0; e < nEpochs; e++) {
        totalLoss = 0;

        while (true) {
            // get the next mini-batch
            Pair<DoubleMatrix, DoubleMatrix> batch = fromBatch(traindata.getNextMiniBatch());
            if (batch == null)
                break;

            // always reset the gradients before performing backward
            optimizer.resetGradients();

            // calculate the loss value
            DoubleMatrix Yhat = net.forward(batch.first);
            double lossVal = loss.forward(batch.second, Yhat);

            // calculate gradients of the weights using backprop algorithm
            net.backward(loss.backward());

            // update the weights using the calculated gradients
            optimizer.updateWeights();

            totalLoss += lossVal;
        }

        // evaluate and print performance
        double trainAcc = eval(net, traindata);
        double valAcc = eval(net, devdata);
        System.out.printf("epoch: %4d\tloss: %5.4f\ttrain-accuracy: %3.4f\tdev-accuracy: %3.4f\n", e, totalLoss, trainAcc, valAcc);

        // check termination condition
        if (valAcc <= peakAcc) {
            notAtPeak += 1;
            System.out.printf("not at peak %d times consecutively\n", notAtPeak);
        }
        else {
            notAtPeak = 0;
            peakAcc = valAcc;
        }
        if (notAtPeak == patience)
            break;
    }

    System.out.println("\ntraining is finished");
}
```

## Reproducibility

To make sure that we can reproduce the results, especially when we need to debug our implementation, we can pass a random seed into our program, and set all relevant random generator with that seed.
There are two random generators in our code:

- `jblas` random generator:
```
org.jblas.util.Random.seed(seed);
```

- `Dataset` random generator: for shuffling the training set during the training process. See `minet.data.Dataset` for details.

## Exercise 

1. Try different network architectures and hyper-parameters (e.g. the number of hidden layers, 
activation functions, the loss function, the learning rate).

1. Select a dataset from [here](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)
and build a neural network. 
*You might need to convert the data to the correct format that
mentioned above.*
