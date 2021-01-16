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
     * Compute a loss value given ground-truth Y and estimate Yhat
     * @param Y a [minibatch_size x d] matrix, each row is the ground-truth of an instance
     * @param Yhat a [minibatch_size x d] matrix, each row is an estimate of an instance
     * @return a double
     */
    @Override
    public double forward(DoubleMatrix Y, DoubleMatrix Yhat) {
    	// if being used for classification, the ground-truth is a vector, so we need to make it become a matrix    	
    	if ((Y.columns==1) && Yhat.columns>1) {    		
    		this.Y = DoubleMatrix.zeros(Yhat.rows, Yhat.columns);    		
            for (int i = 0; i < Y.rows; i++) {
                this.Y.put(i, (int)Y.get(i,0), 1);
            }
    	} else {
    		this.Y = Y.dup();
    	}
        this.Yhat = Yhat.dup();
        return MatrixFunctions.powi(this.Y.sub(Yhat), 2.).columnSums().sum() / this.Y.rows;
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
