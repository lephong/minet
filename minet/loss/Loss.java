// File: Loss.java
// An interface for all loss functions.

package minet.loss;

import org.jblas.*;

/**
 * An interface for all loss functions.
 * @author Phong Le
 */
public interface Loss {

    /**
     * Compute a loss value given groud-truth Y and estimate Yhat
     * @param Y a minibatch_size-row matrix, each row is the ground-truth of an instance
     * @param Yhat a minibatch_size-row matrix, each row an estimate of an instance
     * @return a double
     */
    public double forward(DoubleMatrix Y, DoubleMatrix Yhat);

    /**
     * Compute dL/dYhat
     * @return a minibatch_size-row matrix
     */
    public DoubleMatrix backward();
}
