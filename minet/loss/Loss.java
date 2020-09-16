package minet.loss;

import org.jblas.*;

public interface Loss {
    public DoubleMatrix backward();

    public double getLossVal();
}
