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
public class Sigmoid implements Layer, java.io.Serializable {


	private static final long serialVersionUID = 6451753225913516539L;
	
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
    public List<DoubleMatrix> getAllWeights(List<DoubleMatrix> weights) {
        return weights;
    } 

    @Override
    public List<DoubleMatrix> getAllGradients(List<DoubleMatrix> grads) {
        return grads;
    }

    @Override
    public String toString() {
        return "Sigmoid";
    }
}
