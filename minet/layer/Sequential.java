package minet.layer;

import minet.loss.*;
import org.jblas.DoubleMatrix;

import java.util.LinkedList;
import java.util.List;
import java.util.ListIterator;

public class Sequential implements Layer {
    public Layer[] layers;
    public Loss loss;

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
    public List<double[]> getAllWeights(List<double[]> weights) {
        for (int i = 0; i < layers.length; i++) {
            layers[i].getAllWeights(weights);
        }
        return weights;
    }

    @Override
    public List<double[]> getAllGradients(List<double[]> grads) {
        for (int i = 0; i < layers.length; i++) {
            layers[i].getAllGradients(grads);
        }
        return grads;
    }

    @Override
    public void resetGradients() {
        for (int i = 0; i < layers.length; i++) {
            layers[i].resetGradients();
        }
    }
}
