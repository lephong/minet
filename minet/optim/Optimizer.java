// File: Optimizer.java
// Optimizer class

package minet.optim;

import org.jblas.DoubleMatrix;

/**
 * An interface of optimizers (e.g. {@link SGD}).
 * An optimizer object contains:
 * 1) a list of weight matrices and bias vectors, and
 * 2) a list of their gradients.
 * @author Phong Le
 */
public interface Optimizer {

    /**
     * Set all gradients to 0. This must be called before each update.
     */
    public void resetGradients();

    /**
     * Update parameters using the gradients computed by {@link minet.layer.Layer#backward(DoubleMatrix)}.
     */
    public void updateWeights();
}
