package minet.loss;

import org.jblas.DoubleMatrix;
import org.jblas.MatrixFunctions;

public class MeanSquaredError implements Loss {
    DoubleMatrix Y;
    DoubleMatrix Yhat;

    public MeanSquaredError() { }

    @Override
    public double forward(DoubleMatrix Y, DoubleMatrix Yhat) {
        this.Y = Y.dup();
        this.Yhat = Yhat.dup();
        return MatrixFunctions.powi(Y.sub(Yhat), 2.).columnSums().sum() / Y.rows;
    }

    @Override
    public DoubleMatrix backward() {
        return (this.Y.sub(this.Yhat)).muli(2. / (double) this.Y.rows).muli(-1);
    }

    @Override
    public String toString() {
        return "MeanSquareErrorLoss";
    }
}
