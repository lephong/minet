package minet.layer;

import minet.loss.CrossEntropy;
import org.jblas.*;

import java.util.List;

public class Softmax implements Layer {

    // for backward
    DoubleMatrix Y;
    
    public Softmax() {}

    @Override
    public DoubleMatrix forward(DoubleMatrix X) {
        // Y[:,j] = exp(Y[:,j]) / sum_k exp(Y[:,k])
        DoubleMatrix maxVal = X.rowMaxs();
        DoubleMatrix Y = MatrixFunctions.expi(X.subColumnVector(maxVal));
        DoubleMatrix norm = Y.rowSums();
        Y = Y.diviColumnVector(norm);
        this.Y = Y.dup();
        return Y;
    }

    @Override
    public DoubleMatrix backward(DoubleMatrix gY) {
        // gX[:,j] = Y[:,j] * (gY[:,j] - sum_i gY[:,i] Y[:,i])
        return gY.subColumnVector(this.Y.mul(gY).rowSums()).muli(this.Y);
    }

    @Override
    public List<DoubleMatrix> getAllWeights(List<DoubleMatrix> weights) {
        return weights;
    }

    @Override
    public List<DoubleMatrix> getAllGradients(List<DoubleMatrix> grads) {
        return grads;
    }

    @Override
    public Layer clone() {
        return new Softmax();
    }

    @Override
    public String toString() {
        return "Softmax";
    }
}
