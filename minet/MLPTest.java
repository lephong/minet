package minet;

import minet.layer.*;
import minet.loss.CrossEntropy;
import minet.loss.Loss;
import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class MLPTest {

    public static void train(Layer net, Loss loss, DoubleMatrix X, DoubleMatrix Y) {

    }

    public static void main(String[] args) {
        // create data
        int xDims = 10;
        int nPoints = 1000;
        DoubleMatrix X1 = DoubleMatrix.randn(xDims, (int) nPoints/2).addi(0.5);
        DoubleMatrix Y1 = DoubleMatrix.ones((int) nPoints/2);
        DoubleMatrix X2 = DoubleMatrix.randn(xDims, (int) nPoints/2);
        DoubleMatrix Y2 = DoubleMatrix.zeros((int) nPoints/2);

        DoubleMatrix X = DoubleMatrix.concatVertically(X1, X2);
        DoubleMatrix Y = DoubleMatrix.concatVertically(Y1, Y2);

        // create network
        Sequential net = new Sequential(new Layer[] {
                new Linear(xDims, 2 * xDims, new Linear.WeightInitUniform(-1, 1)),
                new Sigmoid(),
                new Linear(2 * xDims, 2, new Linear.WeightInitUniform(-1, 1)),
                new Softmax()});
        CrossEntropy loss = new CrossEntropy();

        // train network
    }
}
