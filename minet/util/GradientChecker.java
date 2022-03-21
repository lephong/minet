// File: GradientChecker.java
// GradientChecker class
package minet.util;

import minet.layer.*;
import minet.layer.init.*;
import minet.loss.CrossEntropy;
import minet.loss.Loss;
import minet.loss.MeanSquaredError;

import org.jblas.DoubleMatrix;

import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

/**
 * GradientChecker class. This class is to make sure that backward functions
 * (e.g. {@link Linear#backward(DoubleMatrix)}
 * correctly compute gradients.
 * See <a href="http://ufldl.stanford.edu/tutorial/supervised/DebuggingGradientChecking/">this</a>.
 * @author Phong Le
 */
public class GradientChecker {

    /**
     *
     * @param net a neural network
     * @param loss a loss function
     * @param input a dataset of input with minibatch_size items
     * @param Y a minibatch_size-row matrix which is the ground-truth of X.
     */
    public static void checkGradient(Layer net, Loss loss, Object input, DoubleMatrix Y) {
        /* forward and backward to compute the gradients r.t. X and Y */
        loss.forward(Y, net.forward(input));
        net.backward(loss.backward());
        double eps = 1e-7;

        /* checking that dL/dW is computed correctly */
        boolean pass = true;
        List<DoubleMatrix> weights = net.getAllWeights(new LinkedList<DoubleMatrix>());
        ListIterator<DoubleMatrix> wIter = weights.listIterator();

        List<DoubleMatrix> grads = net.getAllGradients(new LinkedList<DoubleMatrix>());
        ListIterator<DoubleMatrix> gIter = grads.listIterator();

        while (wIter.hasNext() && gIter.hasNext()) {
            DoubleMatrix w = wIter.next();
            DoubleMatrix g = gIter.next();

            for (int i = 0; i < w.length; i++) {
                w.data[i] += eps;
                double pLoss = loss.forward(Y, net.forward(input));
                w.data[i] -= 2 * eps;
                double nLoss = loss.forward(Y, net.forward(input));
                w.data[i] += eps;

                double diff = Math.abs(g.data[i] - (pLoss - nLoss) / (2 * eps));
                if (diff > 1e-6) {
                    pass = false;
                    break;
                }
            }
        }

        if (pass)
            System.out.println("correct backward for weights");
        else
            System.err.println("incorrect backward for weights");
    }

    /**
     * Create a classification test.
     */
    public static void testClasification() {
        DoubleMatrix X = new DoubleMatrix(
                new double[][] {
                        {.1f, .1f, .1f, .6f, .1f},
                        {.5f, .1f, .2f, .1f, .1f},
                        {.1f, .2f, .2f, .1f, .4f}});
        DoubleMatrix Y = new DoubleMatrix(new double[] {2., 0., 1.});
        Sequential net = new Sequential(new Layer[] {
                new Linear(5, 10, new WeightInitUniform(-1, 1)),
                new Sigmoid(),
                new Linear(10, 20, new WeightInitUniform(-1, 1)),
                new ReLU(),
                new Linear(20, 6, new WeightInitUniform(-1, 1)),
                new Softmax()});
        CrossEntropy loss = new CrossEntropy();

        System.out.println(net);
        System.out.println(loss);
        checkGradient(net, loss, X, Y);
    }

    public static void main(String[] args) {
        System.out.println("--- Test Classification ---");
        testClasification();

    }

}
