package minet.layer;

import org.jblas.*;

public interface Layer {

    public DoubleMatrix forward(DoubleMatrix X);

    public DoubleMatrix backward(DoubleMatrix gY);
}
