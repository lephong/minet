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

    public static void train(Layer net, Loss loss, Optimizer optimizer, Dataset traindata,
                             Dataset devdata, int batchsize, int nEpochs, int patience) {
        int notAtPeak = 0;
        double peakAcc = -1;
        double totalLoss = 0;

        for (int e = 0; e < nEpochs; e++) {
            System.out.printf("------------ epoch %d ----------\n", e);
            traindata.shuffle();
            totalLoss = 0;

            while (true) {
                Pair<DoubleMatrix> batch = traindata.getNextMiniBatch(batchsize);
                if (batch == null)
                    break;

                optimizer.resetGradients();
                DoubleMatrix Yhat = net.forward(batch.first);
                double lossVal = loss.forward(batch.second, Yhat);
                net.backward(loss.backward());
                optimizer.updateWeights();

                System.out.printf("loss: %f\r", lossVal);
                totalLoss += lossVal;
            }
            System.out.printf("total loss: %f\n", totalLoss);

            double acc = eval(net, devdata, batchsize);
            if (acc < peakAcc) {
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

    public static void main(String[] args) throws IOException {
        double learningRate = 1;
        int batchsize = 100;
        int nEpochs = 100;
        int patience = 5;

        // generate datasets
        System.out.println("========= loading data ========");
        if (args.length == 0) {
            args = new String[] {
                    "/Users/lphong/workspace/minet/data/mnist/mnist_train.txt",
                    "/Users/lphong/workspace/minet/data/mnist/mnist_dev.txt",
                    "/Users/lphong/workspace/minet/data/mnist/mnist_test.txt"
            };
        }
        Dataset trainset = Dataset.loadTxt(args[0]);
        Dataset devset = Dataset.loadTxt(args[1]);
        Dataset testset = Dataset.loadTxt(args[2]);
        System.out.printf("train: %d instances\n", trainset.getSize());
        System.out.printf("dev: %d instances\n", devset.getSize());
        System.out.printf("test: %d instances\n", testset.getSize());

        // create network
        System.out.println("========== creating network ===========");
        int indims = trainset.getInputDims();
        int hiddims = 1000;
        int outdims = 10;
        Sequential net = new Sequential(new Layer[] {
                new Linear(indims, hiddims, new Linear.WeightInitXavier()),
                new ReLU(),
                new Linear(hiddims, outdims, new Linear.WeightInitXavier()),
                new Softmax()});
        CrossEntropy loss = new CrossEntropy();
        Optimizer sgd = new SGD(net, learningRate);
        System.out.println(net);
        System.out.println(loss);

        // train network
        System.out.println("=========== training ===========");
        train(net, loss, sgd, trainset, devset, batchsize, nEpochs, patience);

        // perform on test set
        System.out.println("============ test ============");
        eval(net, testset, batchsize);
    }
}
