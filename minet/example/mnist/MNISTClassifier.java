package minet.example.mnist;

import minet.layer.*;
import minet.loss.CrossEntropy;
import minet.loss.Loss;
import minet.optim.Optimizer;
import minet.optim.SGD;
import minet.util.Pair;

import org.jblas.DoubleMatrix;
import org.jblas.util.Logger;

import java.io.IOException;
import java.util.List;
import java.util.Random;

import minet.layer.init.*;

public class MNISTClassifier {

    /**
     * Convert a mini-batch of MNIST dataset to data structure that can be used by the network
     * @param batch a list of MNIST items, each of which is a pair of (input image, output label)
     * @return two DoubleMatrix objects: X (input) and Y (labels)
    */
    public static Pair<DoubleMatrix, DoubleMatrix> fromBatch(List<Pair<double[], Integer>> batch) {
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

	/** 
     * calculate classification accuracy of an ANN on a given dataset.
     * @param net an ANN model
	 * @param data an MNIST dataset	 
     * @return the classification accuracy value (double, in the range of [0,1])
    */
    public static double eval(Layer net, MNISTDataset data) {
        // reset index of the data
        data.reset();
        
        // the number of correct predictions so far
        double correct = 0;

        while (true) {
            // we evaluate per mini-batch
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

	/** 
     * train an ANN for MNIST
     * @param net an ANN model to be trained
	 * @param loss a loss function object
	 * @param optimizer the optimizer used for updating the model's weights (currently only SGD is supported)
	 * @param traindata training dataset
	 * @param devdata validation dataset (also called development dataset), used for early stopping
	 * @param nEpochs the maximum number of training epochs
	 * @param patience the maximum number of consecutive epochs where validation performance is allowed to non-increased, used for early stopping
    */
    public static void train(Layer net, Loss loss, Optimizer optimizer, MNISTDataset traindata,
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

    
    public static void main(String[] args) throws IOException {
        if (args.length < 4){
            System.out.println("Usage: java MNISTClassifier <seed> <traindata> <devdata> <testdata>");
            return;
        }        

        // set jblas random seed (for reproducibility)		
		org.jblas.util.Random.seed(Integer.parseInt(args[0]));
		Random rnd = new Random(Integer.parseInt(args[0]));
		
        // turn off jblas info messages
        Logger.getLogger().setLevel(Logger.WARNING);

        double learningRate = 0.1;
        int batchsize = 128;
        int nEpochs = 100;
        int patience = 5;
        int hiddims = 500;
        
        // load datasets
        System.out.println("\nLoading data...");
        MNISTDataset trainset = new MNISTDataset(batchsize, true, rnd); 
        trainset.fromFile(args[1]);
        MNISTDataset devset = new MNISTDataset(batchsize, false, rnd); 
        devset.fromFile(args[2]);
        MNISTDataset testset = new MNISTDataset(batchsize, false, rnd); 
        testset.fromFile(args[3]);

        System.out.printf("train: %d instances\n", trainset.getSize());
        System.out.printf("dev: %d instances\n", devset.getSize());
        System.out.printf("test: %d instances\n", testset.getSize());

        // create a network
        System.out.println("\nCreating network...");
        int indims = trainset.getInputDims();
        int outdims = 10;
        Sequential net = new Sequential(new Layer[] {
                new Linear(indims, hiddims, new WeightInitXavier()),
                new ReLU(),
                new Linear(hiddims, outdims, new WeightInitXavier()),
                new Softmax()});
        CrossEntropy loss = new CrossEntropy();
        Optimizer sgd = new SGD(net, learningRate);
        System.out.println(net);

        // train network
        System.out.println("\nTraining...");
        train(net, loss, sgd, trainset, devset, nEpochs, patience);

        // perform on test set
        double testAcc = eval(net, testset);
        System.out.printf("\nTest accuracy: %.4f\n", testAcc);
    }
}
