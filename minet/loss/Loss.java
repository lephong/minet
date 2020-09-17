package minet.loss;

import org.jblas.*;

public interface Loss {

    public double forward(DoubleMatrix Y, DoubleMatrix Yhat);

    public DoubleMatrix backward();
}
