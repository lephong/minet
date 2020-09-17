package minet;

import minet.layer.*;
import minet.loss.CrossEntropy;
import org.jblas.DoubleMatrix;

import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

public class GradientChecking {

    public static void main(String[] args) {
        DoubleMatrix X = new DoubleMatrix(
                new double[][] {
                        {.1f, .1f, .1f, .6f, .1f},
                        {.5f, .1f, .2f, .1f, .1f},
                        {.1f, .2f, .2f, .1f, .4f}});
        int[] labels = new int[] {2, 0, 1};
        Sequential net = new Sequential(new Layer[] {
                new Linear(5, 10, new Linear.WeightInitUniform(-1, 1)),
                new Sigmoid(),
                new Linear(10, 20, new Linear.WeightInitUniform(-1, 1)),
                new ReLU(),
                new Linear(20, 6, new Linear.WeightInitUniform(-1, 1)),
                new Softmax()});
        CrossEntropy loss = new CrossEntropy(labels, net.forward(X));
        DoubleMatrix dX = net.backward(loss.backward());

        // check gradient (input)
        double eps = 1e-7;
        boolean pass = true;

        for (int i = 0; i < X.rows; i++) {
            for (int j = 0; j < X.columns; j++) {
                CrossEntropy pLoss = new CrossEntropy(labels,
                        net.forward(X.dup().put(i, j, X.get(i, j) + eps)));
                CrossEntropy nLoss = new CrossEntropy(labels,
                        net.forward(X.dup().put(i, j, X.get(i, j) - eps)));

                double diff = Math.abs(dX.get(i, j) - (pLoss.lossVal - nLoss.lossVal) / (2 * eps));
                if (diff > 1e-6) {
                    pass = false;
                    break;
                }
            }
        }

        if (pass)
            System.out.println("correct backward for input");
        else
            System.err.println("incorrect forward for input");

        // check gradient (weights)
        pass = true;
        List<double[]> weights = net.getAllWeights(new LinkedList<double[]>());
        ListIterator<double[]> wIter = weights.listIterator();

        List<double[]> grads = net.getAllGradients(new LinkedList<double[]>());
        ListIterator<double[]> gIter = grads.listIterator();

        while (wIter.hasNext() && gIter.hasNext()) {
            double[] w = wIter.next();
            double[] g = gIter.next();

            for (int i = 0; i < w.length; i++) {
                w[i] += eps;
                CrossEntropy pLoss = new CrossEntropy(labels, net.forward(X));
                w[i] -= 2 * eps;
                CrossEntropy nLoss = new CrossEntropy(labels, net.forward(X));
                w[i] += eps;

                double diff = Math.abs(g[i] - (pLoss.lossVal - nLoss.lossVal) / (2 * eps));
                if (diff > 1e-6) {
                    pass = false;
                    break;
                }
            }
        }

        if (pass)
            System.out.println("correct backward for weights");
        else
            System.err.println("incorrect forward for weights");
    }

}
