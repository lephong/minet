package minet.layer;

import org.jblas.DoubleMatrix;

import java.util.List;

public class Sequential implements Layer {
    Layer[] layers;

    public Sequential(Layer[] layers) {
        this.layers = layers;
    }

    @Override
    public Layer clone() {
        Sequential newSeq = new Sequential(new Layer[this.layers.length]);
        for (int i = 0; i < this.layers.length; i++) {
            newSeq.layers[i] = layers[i].clone();
        }
        return newSeq;
    }

    @Override
    public DoubleMatrix forward(DoubleMatrix X) {
        for (int i = 0; i < layers.length; i++) {
            X = layers[i].forward(X);
        }
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
