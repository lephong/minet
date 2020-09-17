package minet.layer;

import org.jblas.*;

import java.util.List;

public interface Layer {

    public DoubleMatrix forward(DoubleMatrix X);

    public DoubleMatrix backward(DoubleMatrix gY);

    public List<DoubleMatrix> getAllWeights(List<DoubleMatrix> weights);

    public List<DoubleMatrix> getAllGradients(List<DoubleMatrix> grads);

    public Layer clone();
}
