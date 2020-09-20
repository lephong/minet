// File: MeanSquaredError.java
// MeanSquaredError class
package minet.loss;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;


/**
 * A class for computing mean squared error loss
 * @author Phong Le
 */
public class MeanSquaredError implements Loss {
    DoubleMatrix Y;
    DoubleMatrix Yhat;

    public MeanSquaredError() { }

    /**
     * Compute a loss value given groud-truth Y and estimate Yhat
     * @param Y a [minibatch_size x d] matrix, each row is the ground-truth of an instance
     * @param Yhat a [minibatch_size x d] matrix, each row is an estimate of an instance
     * @return a double
     */
    @Override
    public double forward(DoubleMatrix Y, DoubleMatrix Yhat) {
        this.Y = Y.dup();
        this.Yhat = Yhat.dup();
        return MatrixFunctions.powi(Y.sub(Yhat), 2.).columnSums().sum() / Y.rows;
    }

    @Override
    public DoubleMatrix backward() {
        return (this.Y.sub(this.Yhat)).muli(2. / (double) this.Y.rows).muli(-1);
    }

    @Override
    public String toString() {
        return "MeanSquareErrorLoss";
    }
}
