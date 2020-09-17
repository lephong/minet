package minet.layer;

import org.jblas.*;

import java.util.List;

public class Sigmoid implements Layer {

    // for backward
    DoubleMatrix Y;
    
    public Sigmoid() {}

    @Override
    public DoubleMatrix forward(DoubleMatrix X) {
        // Y[i] = 1 / (1 + exp(-X[i]))
        DoubleMatrix Y = MatrixFunctions.expi(X.mul(-1)).addi(1).rdivi(1);
        this.Y = Y.dup();
        return Y;
    }

    @Override
    public DoubleMatrix backward(DoubleMatrix gY) {
        // gX = gY . (Y . (1 - Y))
        return gY.mul(this.Y.mul(this.Y.rsub(1)));
    }

    @Override
    public List<double[]> getAllWeights(List<double[]> weights) {
        return weights;
    }

    @Override
    public List<double[]> getAllGradients(List<double[]> grads) {
        return grads;
    }

    @Override
    public Layer clone() {
        return new Sigmoid();
    }

    @Override
    public void resetGradients() { }

}
