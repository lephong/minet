// File: Layer.java
// Interface for all layers.
package minet.layer;

import org.jblas.*;

import java.util.List;

/**
 * This is an interface for all layers
 *
 * @author Phong Le
 */

public interface Layer {

    /**
     * computing the output of applying the layer to the input.
     * @param X is a DoubleMatrix [minibatch_size x input_dims]
     * @return the resulting of applying the layer to input X [minibatch_size x output_dims]
     */
    public DoubleMatrix forward(DoubleMatrix X);

    /**
     * computing the gradient of the layer's parameters and
     * @param gY is a DoubleMatrix
     * @return the resulting of applying backward to gY
     */
    public DoubleMatrix backward(DoubleMatrix gY);

    /**
     * collect all weight matrices and bias vectors
     * @param weights is list DoubleMatrix (is updated accordingly)
     * @return a list of DoubleMatrix
     */
    public List<DoubleMatrix> getAllWeights(List<DoubleMatrix> weights);

    /**
     * collect the gradients of all weight matrices and bias vectors
     * @param grads is list DoubleMatrix (is updated accordingly)
     * @return a list of DoubleMatrix
     */
    public List<DoubleMatrix> getAllGradients(List<DoubleMatrix> grads);

}
