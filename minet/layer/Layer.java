package minet.layer;

import org.jblas.*;

import java.util.List;

public interface Layer {

    public DoubleMatrix forward(DoubleMatrix X);

    public DoubleMatrix backward(DoubleMatrix gY);

    public List<double[]> getAllWeights(List<double[]> weights);

    public List<double[]> getAllGradients(List<double[]> grads);

    public Layer clone();

    public void resetGradients();
}
