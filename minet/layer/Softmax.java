package minet.layer;

import minet.loss.CrossEntropy;
import org.jblas.*;

public class Softmax implements Layer {

    // for backward
    DoubleMatrix Y;
    
    public Softmax() {}

    @Override
    public DoubleMatrix forward(DoubleMatrix X) {
        // Y[i,j] = exp(Y[i,j]) / sum_k exp(Y[i,j])
        DoubleMatrix maxVal = X.rowMaxs();
        DoubleMatrix Y = MatrixFunctions.expi(X.subColumnVector(maxVal));
        DoubleMatrix norm = Y.rowSums();
        Y = Y.diviColumnVector(norm);
        this.Y = Y.dup();
        return Y;
    }

    @Override
    public DoubleMatrix backward(DoubleMatrix gY) {
        DoubleMatrix gX = this.Y.mulColumnVector(this.Y.rowSums().rsubi(1));
        return gX.muli(gY);
    }


}
