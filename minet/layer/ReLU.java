package minet.layer;

import org.jblas.*;

import java.util.List;

public class ReLU implements Layer {

    // for backward
    DoubleMatrix X; 
    
    public ReLU() {}

    @Override
    public DoubleMatrix forward(DoubleMatrix X) {
        this.X = X.dup();

        // Y[i] = max(0, X[i])
        DoubleMatrix Y = X.dup();
        for (int i = 0; i < Y.rows; i++) {
            for (int j = 0; j < Y.columns; j++) {
                if (X.get(i, j) <= 0)
                    Y.put(i, j, 0);
            }
        }
        return Y;
    }

    @Override
    public DoubleMatrix backward(DoubleMatrix gY) {
        // gX[i] = 0 if X[i] <= 0 0 else gY[i]
        DoubleMatrix gX = gY.dup();
        for (int i = 0; i < gX.rows; i++) {
            for (int j = 0; j < gX.columns; j++) {
                if (this.X.get(i, j) <= 0)
                    gX.put(i, j, 0);
            }
        }

        return gX;
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
        return new ReLU();
    }

    @Override
    public String toString() {
        return "ReLU";
    }
}
