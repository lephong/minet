// File: CrossEntropy.java
// CrossEntropy class
package minet.loss;

import org.jblas.DoubleMatrix;

/**
 * A class for computing cross entropy loss
 * @author Phong Le
 */
public class CrossEntropy implements Loss {
    DoubleMatrix Yhat;
    int[] labels;

    /**
     * Constructor for CrossEntropy loss function
     */
    public CrossEntropy() { }

    /**
     * Compute a loss value given groud-truth Y and estimate Yhat
     * @param Y a [minibatch_size x 1] matrix, each row is the ground-truth label of an instance
     * @param Yhat a [minibatch_size x d] matrix, each row is a distribution over the category set
     * @return the loss value (a double)
     */
    @Override
    public double forward(DoubleMatrix Y, DoubleMatrix Yhat) {
        this.Yhat = Yhat.dup();
        this.labels = new int[Y.length];
        for (int i = 0; i < Y.length; i++) {
            this.labels[i] = (int) Y.data[i];
        }

        double lossVal = 0;
        for (int i = 0; i < labels.length; i++) {
            lossVal -= Math.log(Yhat.get(i, labels[i]) + 1e-7);
        }
        return lossVal / (double)labels.length;
    }

    /**
     * Compute gradient of the loss wrt output nodes of the network
     * @return a matrix (DoubleMatrix), with #rows = #samples, #cols = #output nodes of the network
     */
    @Override
    public DoubleMatrix backward() {
        DoubleMatrix dY = DoubleMatrix.zeros(this.Yhat.rows, this.Yhat.columns);
        for (int i = 0; i < this.labels.length; i++) {
            dY.put(i, this.labels[i], -1 / (this.Yhat.get(i, this.labels[i]) + 1e-7));
        }
        return dY.divi(dY.rows);
    }

    @Override
    public String toString() {
        return "CrossEntropyLoss";
    }
}
