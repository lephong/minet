package minet.example;

import minet.Dataset;
import minet.layer.*;
import minet.loss.CrossEntropy;
import minet.loss.Loss;
import minet.optim.Optimizer;
import minet.optim.SGD;
import minet.util.Pair;
import org.jblas.DoubleMatrix;

import java.io.IOException;


public class MNISTClassifier {

    public static Dataset[] loadDatasets(String train, String dev, String test) throws IOException {
        Dataset[] datasets = new Dataset[] {
                Dataset.loadTxt(train),
                Dataset.loadTxt(dev),
                Dataset.loadTxt(test)};
        return datasets;
    }

    public static double eval(Layer net, Dataset data, int batchsize) {
        data.reset();
        double correct = 0;

        while (true) {
            Pair<DoubleMatrix> batch = data.getNextMiniBatch(batchsize);
            if (batch == null)
                break;

            DoubleMatrix Yhat = net.forward(batch.first);
            int[] preds = Yhat.rowArgmaxs();
            for (int i = 0; i < preds.length; i++) {
                if (preds[i] == (int) batch.second.data[i])
                    correct++;
            }
        }
        double acc = correct / data.getSize();
        System.out.printf("accuracy %f \n", acc);
        return acc;
    }

    public static void train(Layer net, Loss loss, Optimizer optimizer,
                             Dataset traindata, Dataset devdata,
                             int batchsize, int nEpochs) {
        for (int e = 0; e < nEpochs; e++) {
            System.out.printf("------------ epoch %d ----------\n", e);
            traindata.shuffle();

            while (true) {
                Pair<DoubleMatrix> batch = traindata.getNextMiniBatch(batchsize);
                if (batch == null)
                    break;

                optimizer.resetGradients();
                double lossVal = loss.forward(batch.second, net.forward(batch.first));
                net.backward(loss.backward());
                optimizer.updateWeights();

                System.out.printf("loss: %f\r", lossVal);
            }
            System.out.printf("\n");
            double acc = eval(net, devdata, batchsize);
        }
    }

    public static void main(String[] args) throws IOException {
        double learningRate = 1;
        int batchsize = 100;
        int nEpochs = 100;

        // generate datasets
        System.out.println("loading data...");
        Dataset trainset = Dataset.loadTxt("/Users/lphong/workspace/minet/data/mnist/mnist_train.txt");
        Dataset devset = Dataset.loadTxt("/Users/lphong/workspace/minet/data/mnist/mnist_test.txt");
        Dataset testset = Dataset.loadTxt("/Users/lphong/workspace/minet/data/mnist/mnist_test.txt");

        // create network
        System.out.println("creating network...");
        int indims = trainset.getInputDims();
        int outdims = 10;
        Sequential net = new Sequential(new Layer[] {
                new Linear(indims, (int) 1.5 * indims, new Linear.WeightInitXavier()),
                new ReLU(),
                new Linear((int) 1.5 * indims, outdims, new Linear.WeightInitXavier()),
                new Softmax()});
        CrossEntropy loss = new CrossEntropy();
        Optimizer sgd = new SGD(net, learningRate);
        System.out.println(net);
        System.out.println(loss);

        // train network
        System.out.println("training...");
        train(net, loss, sgd, trainset, devset, batchsize, nEpochs);
    }
}
