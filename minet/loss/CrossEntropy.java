package minet.loss;

import org.jblas.DoubleMatrix;

public class CrossEntropy implements Loss {
    DoubleMatrix Yhat;
    int[] labels;

    public CrossEntropy() { }

    @Override
    public double forward(DoubleMatrix Y, DoubleMatrix Yhat) {
        this.Yhat = Yhat.dup();
        this.labels = new int[Y.length];
        for (int i = 0; i < Y.length; i++) {
            this.labels[i] = (int) Y.data[i];
        }

        double lossVal = 0;
        for (int i = 0; i < labels.length; i++) {
            lossVal -= Math.log(Yhat.get(i, labels[i]));
        }
        return lossVal / (double)labels.length;
    }

    @Override
    public DoubleMatrix backward() {
        DoubleMatrix dY = DoubleMatrix.zeros(this.Yhat.rows, this.Yhat.columns);
        for (int i = 0; i < this.labels.length; i++) {
            dY.put(i, this.labels[i], -1 / this.Yhat.get(i, this.labels[i]));
        }
        return dY.divi(dY.rows);
    }

    @Override
    public String toString() {
        return "CrossEntropyLoss";
    }
}
