// File: SGD.java
// SGD class
package minet.optim;

import minet.layer.Layer;
import org.jblas.DoubleMatrix;

import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;


/**
 * SGD (stochastic gradient descent) class
 * @author Phong Le
 */
public class SGD implements Optimizer {
    List<DoubleMatrix> weights;
    List<DoubleMatrix> grads;
    double lr;

    public SGD(Layer net, double learningRate) {
        this.lr = learningRate;

        this.weights = new LinkedList<DoubleMatrix>();
        this.grads = new LinkedList<DoubleMatrix>();
        net.getAllWeights(this.weights);
        net.getAllGradients(this.grads);
    }

    /**
     * Set learning rate.
     * @param lr a double
     */
    public void setLearningRate(double lr) {
        this.lr = lr;
    }

    @Override
    public void resetGradients() {
        ListIterator<DoubleMatrix> gIter = this.grads.listIterator();
        while (gIter.hasNext()) {
            DoubleMatrix g = gIter.next();
            g.fill(0);
        }
    }

    @Override
    public void updateWeights() {
        ListIterator<DoubleMatrix> wIter = this.weights.listIterator();
        ListIterator<DoubleMatrix> gIter = this.grads.listIterator();
        while (gIter.hasNext() && wIter.hasNext()) {
            DoubleMatrix w = wIter.next();
            DoubleMatrix g = gIter.next();
            w.addi(g.mul(-this.lr));
        }
    }
}
