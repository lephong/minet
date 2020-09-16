package minet.loss;

import org.jblas.DoubleMatrix;

public class CrossEntropy implements Loss {
    public DoubleMatrix Yhat;
    public int[] labels;
    public double lossVal;

    public CrossEntropy(int[] labels, DoubleMatrix Yhat) {
        // Yhat: category distribution p(class i) [batchsize x nClasses]
        assert labels.length == Yhat.rows : "#labels must equal to batch-size";
        this.lossVal = 0;
        for (int i = 0; i < labels.length; i++) {
            this.lossVal -= Math.log(Yhat.get(i, labels[i]));
        }
        this.lossVal = this.lossVal / (double)labels.length;

        this.Yhat = Yhat.dup();
        this.labels = labels.clone();
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
    public double getLossVal() {
        return this.lossVal;
    }

    public static void main(String[] args) {
        DoubleMatrix Yhat = new DoubleMatrix(
                new double[][] {
                        {.1f, .1f, .1f, .6f, .1f},
                        {.5f, .1f, .2f, .1f, .1f},
                        {.1f, .2f, .2f, .1f, .4f}});
        int[] labels = new int[] {2, 0, 1};
        CrossEntropy loss = new CrossEntropy(labels, Yhat);
        DoubleMatrix dY = loss.backward();

        // check forward
        double correctL = (double) (- (Math.log(.1f) + Math.log(.5f) + Math.log(.2f)) / 3.f);
        if (Math.abs(loss.lossVal - correctL) < 1e-6)
            System.out.println("correct forward");
        else
            System.err.println("incorrect forward");

        // check gradient
        double eps = (double) 1e-7;
        boolean pass = true;

        for (int i = 0; i < Yhat.rows; i++) {
            for (int j = 0; j < Yhat.columns; j++) {
                CrossEntropy pLoss = new CrossEntropy(labels,
                        Yhat.dup().put(i, j, Yhat.get(i, j) + eps));
                CrossEntropy nLoss = new CrossEntropy(labels,
                        Yhat.dup().put(i, j, Yhat.get(i, j) - eps));

                if (Math.abs(dY.get(i, j) - (pLoss.lossVal - nLoss.lossVal) / (2 * eps)) > 1e-6) {
                    pass = false;
                    break;
                }
            }
        }

        if (pass)
            System.out.println("correct backward");
        else
            System.err.println("incorrect forward");

    }
}
