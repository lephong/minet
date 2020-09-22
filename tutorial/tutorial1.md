# Tutorial 1: Building a multi-layer neural network (MLNN)

In this tutorial, we will build a handwritten digit recognizer using 
[MNIST dataset](http://yann.lecun.com/exdb/mnist/). 
The source code is available in the file `mnit.example.MNISTClassifier.java`. 
  
## Preparing data
We need a text file with the following format

```
[n = Number of instances] [dx = dims of x] [dy = dims of y]
[dx floats, separated by " "] ; [dy floats, separated by " "]
...
[dx floats, separated by " "] ; [dy floats, separated by " "]
...
```

For instance, we represent the MNIST data as follows 
(pixel values are normalized)

```
60000 784 1 
0.0 ... 0.9921568627450981 0.9921568627450981 ... 0.0 ; 5
...
0.0 ... 0.24313725490196078 0.3176470588235294 ... 0.0 ; 4
...
```
  
There are 60000 images (x) with size 28x28. The first image is digit 5. 
Note that for classification, the dimensionality of y is always 1 regardless to 
the number of categories. 

Assuming that the train/dev/test datasets are stored at 

```
../data/mnist/mnist_train.txt
../data/mnist/mnist_dev.txt
../data/mnist/mnist_test.txt
```

## Loading the data
We use the class `minet.Dataset` to store datasets and 
iterator over mini-batches. 

```java
Dataset trainset = Dataset.loadTxt("../data/mnist/mnist_train.txt");    // 50k images
Dataset devset = Dataset.loadTxt("../data/mnist/mnist_test.txt");       // 10k images
Dataset testset = Dataset.loadTxt("../data/mnist/mnist_test.txt");      // 10k images
```

## Building an MLNN
We build a one-hidden-layer neural network classifier: 
- The input layer has 784 nodes, corresponding to 784 dims of x. 
- The hidden layer has 1000 nodes, with ReLU activation function.
- The output layer has 10 nodes, corresponding to 10 digits. 
Because the task is classification, 
we use a softmax to compute a distribution over 
the 10 digits. 

```java
int indims = trainset.getInputDims();  # get the dims of x (784)
int hiddims = 1000; 
int outdims = 10;  # there are 10 digits (i.e. 10 categories)
Sequential net = new Sequential(new Layer[] {
        new Linear(indims, hiddims, new Linear.WeightInitXavier()),
        new ReLU(),
        new Linear(hiddims, outdims, new Linear.WeightInitXavier()),
        new Softmax()});
```
We use cross-entropy loss to train the network

```java
CrossEntropy loss = new CrossEntropy();
```
    
and stochastic gradient descent to minimize the loss function

```java
double learningRate = 1;
Optimizer sgd = new SGD(net, learningRate);
```

Finally, for logging, 
we print out the structure of the network and the loss 

```java
System.out.println(net);
System.out.println(loss);
```
    
    
## Evaluating the network
For classification, we evaluate the network using accuracy metric
`accuracy = #correct / #total`. Other widely used metrics include 
[precision, recall, and F1](https://en.wikipedia.org/wiki/Precision_and_recall). 

```java
public static double eval(Layer net,
                          Dataset data,
                          int batchsize) {
    data.reset();  # always reset before use 
    double correct = 0;  // for counting how many predictions are correct

    // we process every mini-batch
    while (true) {
        // get the next mini-batch. Each mini-batch is a (first, second) pair
        // whose "first" is images X and "second" is the ground-truth labels Y.
        Pair<DoubleMatrix> batch = data.getNextMiniBatch(batchsize);  
        if (batch == null)  // stop when no items are left
            break;

        // perform forward to compute Yhat, each row of whom is a distribution over 10 digits
        DoubleMatrix Yhat = net.forward(batch.first);

        // the predicted digit for each image is the one with the highest probability
        int[] preds = Yhat.rowArgmaxs();

        // count how many predictions are correct
        for (int i = 0; i < preds.length; i++) {
            if (preds[i] == (int) batch.second.data[i])
                correct++;
        }
    }
    
    // compute accuracy
    double acc = correct / data.getSize();
    System.out.printf("accuracy %f \n", acc);
    return acc;
}
```



## Training the network
We train the network by updating its weights after processing 
each mini-batch. After each epoch, we evaluate the current weights
on a development set. If the accuracy is not at pick `patience` times 
consecutively, we halt the training (this is an 
[early stopping](https://en.wikipedia.org/wiki/Early_stopping)
stragety).

Note:
- always *shuffle* the training data before each epoch,
- always *reset* the gradients before performing backward.

```java
public static void train(Layer net, Loss loss, Optimizer optimizer, Dataset traindata,
                         Dataset devdata, int batchsize, int nEpochs, int patience) {
    int notAtPeak = 0;  // the number of times not at peak
    double peakAcc = -1;  // the accuracy of the previous epoch
    double totalLoss = 0;  // the total loss of the current epoch

    for (int e = 0; e < nEpochs; e++) {
        System.out.printf("------------ epoch %d ----------\n", e);
        traindata.shuffle();  // always shuffle the data before each epoch.
        totalLoss = 0;

        while (true) {
            Pair<DoubleMatrix> batch = traindata.getNextMiniBatch(batchsize);  // get the next mini-batch
            if (batch == null)  // finish this epoch if there are no items left
                break;

            optimizer.resetGradients();  // always reset the gradients before performing backward
            DoubleMatrix Yhat = net.forward(batch.first);
            double lossVal = loss.forward(batch.second, Yhat);
            net.backward(loss.backward());
            optimizer.updateWeights();

            System.out.printf("loss: %f\r", lossVal);
            totalLoss += lossVal;
        }
        System.out.printf("total loss: %f\n", totalLoss);

        // early stopping
        double acc = eval(net, devdata, batchsize);
        if (acc < prevAcc) {
            notAtPeak += 1;
            System.out.printf("not at peak %d times consecutively\n", notAtPeak);
        }
        else {
            notAtPeak = 0;
            peakAcc = acc;
        }
        if (notAtPeak == patience)
            break;
    }
}
```

The output should look like
```
loading data...
creating network...
(
    Linear: 784 in, 1000 out
    ReLU
    Linear: 1000 in, 10 out
    Softmax
)
CrossEntropyLoss
training...
------------ epoch 0 ----------
total loss: 161.838046
accuracy 0.955700 
------------ epoch 1 ----------
total loss: 53.715121
accuracy 0.963900 
------------ epoch 2 ----------
total loss: 34.836546
accuracy 0.965000 
------------ epoch 3 ----------
total loss: 24.944736
accuracy 0.967700 
------------ epoch 4 ----------
total loss: 17.495233
accuracy 0.968100 
------------ epoch 5 ----------
total loss: 12.456065
accuracy 0.972000 
------------ epoch 6 ----------
total loss: 8.450051
accuracy 0.972100 
------------ epoch 7 ----------
total loss: 5.996042
accuracy 0.972900 
------------ epoch 8 ----------
total loss: 3.632146
accuracy 0.972800 
not at peak 1 times consecutively
```

## Exercise 

Select a dataset from [here](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/)
and build a neural network.