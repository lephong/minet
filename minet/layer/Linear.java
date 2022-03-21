// File: Linear.java
// Linear layer
package minet.layer;

import org.jblas.*;

import java.util.List;
import minet.layer.init.*;

/**
 * A class for linear layers (Y = XW + b)
 *
 * @author Phong Le
 */
public class Linear implements Layer, java.io.Serializable {			

	private static final long serialVersionUID = -10435336293457306L;
	
	DoubleMatrix W;  // weight matrix
    DoubleMatrix b;  // bias vector

    // for backward
    DoubleMatrix X;   // store input X for computing backward
    DoubleMatrix gW;  // gradient of W
    DoubleMatrix gb;  // gradient of b

    public Linear(int indims, int outdims, WeightInit wInit) {
        this.W = wInit.generate(indims, outdims);
        this.b = DoubleMatrix.zeros(outdims);
        this.gW = DoubleMatrix.zeros(indims, outdims);
        this.gb = DoubleMatrix.zeros(outdims);
    }

    public Linear(DoubleMatrix W, DoubleMatrix b) {
        this.W = W;
        this.b = b;
    }

    @Override
    public DoubleMatrix forward(Object input) {
        DoubleMatrix X = (DoubleMatrix)input;
        // Y = X * W + b
        DoubleMatrix Y = X.mmul(W).addiRowVector(b);
        this.X = X.dup();
        return Y;
    }

    @Override
    public DoubleMatrix backward(DoubleMatrix gY) {
        // gW = X^T * gY
        gW.addi(X.transpose().mmul(gY));

        // gb = sum_row gY
        gb.addi(gY.columnSums());

        // gX = gY * W^T
        return gY.mmul(W.transpose());
    }

    @Override
    public List<DoubleMatrix> getAllWeights(List<DoubleMatrix> weights) {
        weights.add(W);
        weights.add(b);
        return weights;
    }

    @Override
    public List<DoubleMatrix> getAllGradients(List<DoubleMatrix> grads) {
        grads.add(gW);
        grads.add(gb);
        return grads;
    }

    @Override
    public String toString() {
        return String.format("Linear: %d in, %d out", W.rows, W.columns);
    }
}
