package minet;

import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.util.Random;

public class Dataset {

    int currIndex;

    public class DoubleMatrixPair {
        DoubleMatrix X;
        DoubleMatrix Y;

        public DoubleMatrixPair(DoubleMatrix X, DoubleMatrix Y) {
            this.X = X;
            this.Y = Y;
        }
    }

    DoubleMatrixPair data;

    public Dataset(DoubleMatrix X, DoubleMatrix Y) {
        data = new DoubleMatrixPair(X, Y);
        this.currIndex = 0;
    }

    public void shuffle() {
        Random rnd = new Random();
        for (int i = this.data.X.rows - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            this.data.X.swapRows(index, i);
            this.data.Y.swapRows(index, i);
        }
    }

    public DoubleMatrixPair getNextMiniBatch(int batchsize) {
        int start = this.currIndex;
        int end = Math.min(start + batchsize, this.data.X.rows);
        this.currIndex = end;

        return new DoubleMatrixPair(
                data.X.getRows(new IntervalRange(start, end)),
                data.Y.getRows(new IntervalRange(start, end))
        );
    }
}
