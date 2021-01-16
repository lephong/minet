// File: Sequential.java
// Sequential

package minet.layer;

import org.jblas.DoubleMatrix;

import java.util.List;


/**
 * A sequential container, containing a sequence of layers.
 * For example: a sequential object with the list of layers
 * [Linear, ReLU, Linear, Softmax] is equivalent to
 * {@literal X=X1 -> Linear -> X2 -> ReLU -> X3 -> Linear -> X4 -> Softmax -> Y=X5}
 * @author Phong Le
 */
public class Sequential implements Layer, java.io.Serializable {
	
	private static final long serialVersionUID = 2172439814486831959L;
	
	Layer[] layers;         

    public Sequential(Layer[] layers) {
        this.layers = layers;
    }

    @Override
    public DoubleMatrix forward(DoubleMatrix X) {
    	//System.out.print("\n");
        for (int i = 0; i < layers.length; i++) {
            X = layers[i].forward(X);
            //System.out.print(X.mean() + " ");
        }
        //System.out.print("\n");
        return X;
    }

    @Override
    public DoubleMatrix backward(DoubleMatrix dY) {
        for (int i = layers.length-1; i >= 0; i--) {
            dY = layers[i].backward(dY);
        }
        return dY;
    }

    @Override
    public List<DoubleMatrix> getAllWeights(List<DoubleMatrix> weights) {
        for (int i = 0; i < layers.length; i++) {
            layers[i].getAllWeights(weights);
        }
        return weights;
    }

    @Override
    public List<DoubleMatrix> getAllGradients(List<DoubleMatrix> grads) {
        for (int i = 0; i < layers.length; i++) {
            layers[i].getAllGradients(grads);
        }
        return grads;
    }

    @Override
    public String toString() {
        StringBuilder str = new StringBuilder();
        str.append("(\n");
        for (int i = 0; i < layers.length; i++) {
            str.append("    ").append(layers[i].toString()).append("\n");
        }
        str.append(")");
        return str.toString();
    }
}
