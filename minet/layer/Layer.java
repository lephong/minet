// File: Layer.java
// An interface for all layers.
package minet.layer;

import org.jblas.*;

import java.util.List;

/**
 * An interface for all layers.
 *
 * @author Phong Le
 */
public interface Layer {

    /**
     * Computing the output of applying the layer to input X, i.e. Y = Layer(X).
     * @param X a [minibatch_size x input_dims] matrix, each row is an input instance
     * @return a [minibatch_size x output_dims] matrix, each row is the output of the corresponding instance
     */
    public DoubleMatrix forward(DoubleMatrix X);

    /**
     * Computing the gradient of the layer's parameters and the input
     * when applying {@link forward}.
     * @param gY a [minibatch_size x output_dims] matrix, each row is dL/dY
     * @return a [minibatch_size x input_dims] matrix, each row is dL/dX
     * where L is a loss function (@see {@link minet.loss.Loss}).
     */
    public DoubleMatrix backward(DoubleMatrix gY);

    /**
     * Collect all the weight matrices and bias vectors of the layer.
     * @param weights a list of matrices (updated accordingly)
     * @return the same list.
     */
    public List<DoubleMatrix> getAllWeights(List<DoubleMatrix> weights);      

    /**
     * Collect the gradients of all the weight matrices and bias vectors
     * of the layer.
     * @param grads a list of matrices  (updated accordingly)
     * @return the same list.
     */
    public List<DoubleMatrix> getAllGradients(List<DoubleMatrix> grads);

}
