package minet.layer;

import minet.loss.*;
import org.jblas.DoubleMatrix;

public class Sequential implements Layer {
    public Layer[] layers;
    public Loss loss;

    public Sequential(Layer[] layers) {
        this.layers = layers;
    }

    @Override
    public DoubleMatrix forward(DoubleMatrix X) {
        for (int i = 0; i < layers.length; i++) {
            X = layers[i].forward(X);
        }
        return X;
    }

    @Override
    public DoubleMatrix backward(DoubleMatrix dY) {
        for (int i = layers.length-1; i >= 0; i--) {
            dY = layers[i].backward(dY);
        }
        return dY;
    }

    public static void main(String[] args) {
        DoubleMatrix X = new DoubleMatrix(
                new double[][] {
                        {.1f, .1f, .1f, .6f, .1f},
                        {.5f, .1f, .2f, .1f, .1f},
                        {.1f, .2f, .2f, .1f, .4f}});
        int[] labels = new int[] {2, 0, 1};
        Sequential net = new Sequential(new Layer[] {new Softmax()});
        DoubleMatrix Yhat = net.forward(X);
        CrossEntropy loss = new CrossEntropy(labels, Yhat);
        DoubleMatrix dY = loss.backward();
        DoubleMatrix dX = net.backward(dY);

        // check gradient
        double eps = (double) 1e-7;
        boolean pass = true;

        for (int i = 0; i < X.rows; i++) {
            for (int j = 0; j < X.columns; j++) {
                CrossEntropy pLoss = new CrossEntropy(labels,
                        net.forward(X.dup().put(i, j, X.get(i, j) + eps)));
                CrossEntropy nLoss = new CrossEntropy(labels,
                        net.forward(X.dup().put(i, j, X.get(i, j) - eps)));

                double diff = Math.abs(dX.get(i, j) - (pLoss.lossVal - nLoss.lossVal) / (2 * eps));
                if (diff > 1e-6) {
                    System.out.println(diff);
                    pass = false;
                    break;
                }
            }
        }

        if (pass)
            System.out.println("correct backward");
        else
            System.err.println("incorrect forward");

    }

}
