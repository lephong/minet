// File: Sigmoid.java
// Sigmoid layer
package minet.layer;

import org.jblas.*;

import java.util.List;

/**
 * A class for sigmoid layers {@literal y = 1 / (1 + exp(-x))}.
 *
 * @author Phong Le
 */
public class Tanh implements Layer, java.io.Serializable {

	private static final long serialVersionUID = -7444093094282163781L;
	// for backward
    DoubleMatrix Y;
    
    public Tanh() {}

    @Override
    public DoubleMatrix forward(DoubleMatrix X) {
        // Y[i] = tanh(X[i])
        DoubleMatrix Y = MatrixFunctions.tanh(X);
        this.Y = Y.dup();
        return Y;
    }

    @Override
    public DoubleMatrix backward(DoubleMatrix gY) {
        // gX = gY . (1 - Y * Y)
        return gY.mul((this.Y.mul(this.Y)).rsub(1));
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
    public String toString() {
        return "Tanh";
    }
}
