# Tutorial 1: Building a multi-layer neural network (MLNN)

In this tutorial, we will build a handwritten digit recognizer using 
[MNIST dataset](http://yann.lecun.com/exdb/mnist/). 
The source code is available in the file `mnit.example.MNISTClassifier.java`. 
  
##1. Preparing data
We need a text file with the following format

    [n = Number of instances] [dx = dims of x] [dy = dims of y]
    [dx floats, separated by " "] ; [dy floats, separated by " "]
    ...
    [dx floats, separated by " "] ; [dy floats, separated by " "]
    ...

For instance, we represent the MNIST data as follows 
(pixel values are normalized)

    60000 784 1 
    0.0 ... 0.9921568627450981 0.9921568627450981 ... 0.0 ; 5
    ...
    0.0 ... 0.24313725490196078 0.3176470588235294 ... 0.0 ; 4
    ...
  
There are 60000 images (x) with size 28x28. The first image is digit 5. 
Note that for classification, the dimensionality of y is always 1 regardless to 
the number of categories. 

Assuming that the train/dev/test datasets are stored at 

    ../data/mnist/mnist_train.txt
    ../data/mnist/mnist_dev.txt
    ../data/mnist/mnist_test.txt

##2. Loading the data
We use the class `minet.Dataset` to store datasets and 
iterator over mini-batches. 

    Dataset trainset = Dataset.loadTxt("../data/mnist/mnist_train.txt");
    Dataset devset = Dataset.loadTxt("../data/mnist/mnist_test.txt");
    Dataset testset = Dataset.loadTxt("../data/mnist/mnist_test.txt");

##3. Buiding an MLNN
We build a one-hidden-layer neural network classifier: 
- The input layer has 784 nodes, corresponding to 784 dims of x. 
- The hidden layer has 1176 nodes, with ReLU activation function.
- The output layer has 10 nodes, corresponding to 10 digits. 
Because the task is classification, 
we use a softmax to compute a distribution over 
the 10 digits. 


    int indims = trainset.getInputDims();  # get the dims of x (784)
    int outdims = 10;  # there are 10 digits (i.e. 10 categories)
    Sequential net = new Sequential(new Layer[] {
            new Linear(indims, (int) 1.5 * indims, new Linear.WeightInitXavier()),
            new ReLU(),
            new Linear((int) 1.5 * indims, outdims, new Linear.WeightInitXavier()),
            new Softmax()});
            
We use cross-entropy loss to train the network

    CrossEntropy loss = new CrossEntropy();
    
and stochastic gradient descent to minimize the loss function

    double learningRate = 1;
    Optimizer sgd = new SGD(net, learningRate);

Finally, for logging, 
we print out the structure of the network and the loss 

    System.out.println(net);
    System.out.println(loss);
    
##4. Training the network
Training the network requires to ha

