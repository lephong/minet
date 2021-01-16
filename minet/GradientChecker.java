// File: GradientChecker.java
// GradientChecker class
package minet;

import minet.layer.*;
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
     * @param X a minibatch_size x input_dims matrix
     * @param Y a minibatch_size-row matrix which is the ground-truth of X.
     */
    public static void checkGradient(Layer net, Loss loss, DoubleMatrix X, DoubleMatrix Y) {
        /* forward and backward to compute the gradients r.t. X and Y */
        loss.forward(Y, net.forward(X));
        DoubleMatrix dX = net.backward(loss.backward());

        double eps = 1e-7;

        /* checking that dL/dX is computed correctly */
        boolean pass = true;
        for (int i = 0; i < X.rows; i++) {
            for (int j = 0; j < X.columns; j++) {
                double pLoss = loss.forward(Y,
                        net.forward(X.dup().put(i, j, X.get(i, j) + eps)));
                double nLoss = loss.forward(Y,
                        net.forward(X.dup().put(i, j, X.get(i, j) - eps)));

                double diff = Math.abs(dX.get(i, j) - (pLoss - nLoss) / (2 * eps));
                if (diff > 1e-6) {
                    pass = false;
                    break;
                }
            }
        }

        if (pass)
            System.out.println("correct backward for input");
        else
            System.err.println("incorrect backward for input");

        /* checking that dL/dW is computed correctly */
        pass = true;
        List<DoubleMatrix> weights = net.getAllWeights(new LinkedList<DoubleMatrix>());
        ListIterator<DoubleMatrix> wIter = weights.listIterator();

        List<DoubleMatrix> grads = net.getAllGradients(new LinkedList<DoubleMatrix>());
        ListIterator<DoubleMatrix> gIter = grads.listIterator();

        while (wIter.hasNext() && gIter.hasNext()) {
            DoubleMatrix w = wIter.next();
            DoubleMatrix g = gIter.next();

            for (int i = 0; i < w.length; i++) {
                w.data[i] += eps;
                double pLoss = loss.forward(Y, net.forward(X));
                w.data[i] -= 2 * eps;
                double nLoss = loss.forward(Y, net.forward(X));
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
    public static void test2() {
        DoubleMatrix X = new DoubleMatrix(
                new double[][] {
                        {.1f, .1f, .1f, .6f, .1f},
                        {.5f, .1f, .2f, .1f, .1f},
                        {.1f, .2f, .2f, .1f, .4f}});
        DoubleMatrix Y = new DoubleMatrix(new double[] {2., 0., 1.});
        Sequential net = new Sequential(new Layer[] {
                new Linear(5, 10, new Linear.WeightInitUniform(-1, 1)),
                new Sigmoid(),
                new Linear(10, 20, new Linear.WeightInitUniform(-1, 1)),
                new ReLU(),
                new Linear(20, 6, new Linear.WeightInitUniform(-1, 1)),
                new Softmax()});
        CrossEntropy loss = new CrossEntropy();

        System.out.println(net);
        System.out.println(loss);
        checkGradient(net, loss, X, Y);
    }

    /**
     * Create a regression test.
     */
    public static void test1() {
        DoubleMatrix X = new DoubleMatrix(
                new double[][] {
                        {.1f, .1f, .1f, .6f, .1f},
                        {.5f, .1f, .2f, .1f, .1f},
                        {.1f, .2f, .2f, .1f, .4f}});
        DoubleMatrix Y = new DoubleMatrix(new double[][] {
                {2., 0.},
                {-0.1, 5.},
                {3., -1.2}});
        Sequential net = new Sequential(new Layer[]{
                new Linear(5, 2, new Linear.WeightInitUniform(-1, 1))
        });
        MeanSquaredError loss = new MeanSquaredError();

        System.out.println(net);
        System.out.println(loss);
        checkGradient(net, loss, X, Y);
    }


    public static void main(String[] args) {
        System.out.println("--- Test 1 ---");
        test1();

        System.out.println();
        System.out.println("--- Test 2 ---");
        test2();
    }

}
