// File: Softmax.java
// Softmax layer

package minet.layer;

import org.jblas.*;

import java.util.List;


/**
 * A class for softmax layers {@literal y[i] = exp(x[i]) / sum_j exp(x[j])}.
 *
 * @author Phong Le
 */
public class Softmax implements Layer, java.io.Serializable {	

	private static final long serialVersionUID = 8714215486185502826L;
	
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
    public String toString() {
        return "Softmax";
    }
}
