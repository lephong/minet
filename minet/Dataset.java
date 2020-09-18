package minet;

import minet.util.Pair;
import org.jblas.DoubleMatrix;
import org.jblas.ranges.IntervalRange;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;
import java.util.stream.Stream;

public class Dataset {

    int currIndex;
    double[][] X;
    double[][] Y;

    public Dataset(double[][] X, double[][] Y) {
        this.X = X;
        this.Y = Y;
        this.currIndex = 0;
    }

    public static Dataset loadTxt(String path) throws IOException {
        // first line: [number of samples] [xDims] [yDims]
        // each line of file : [entries of X] ; [entries of Y]
        BufferedReader br = new BufferedReader(new FileReader(path));

        // first line
        String[] ss = br.readLine().split(" ");
        int size = Integer.valueOf(ss[0]);
        int xDims = Integer.valueOf(ss[1]);
        int yDims = Integer.valueOf(ss[2]);

        double[][] X = new double[size][xDims];
        double[][] Y = new double[size][yDims];

        for (int i = 0; i < size; i++) {
            ss = br.readLine().split(" ; ");
            String[] sx = ss[0].split(" ");
            String[] sy = ss[1].split(" ");
            for (int j = 0; j < sx.length; j++) {
                X[i][j] = Double.valueOf(sx[j]);
            }
            for (int j = 0; j < sy.length; j++) {
                Y[i][j] = Double.valueOf(sy[j]);
            }
        }

        return new Dataset(X, Y);
    }

    public void reset() {
        this.currIndex = 0;
    }

    public int getSize() {
        return X.length;
    }

    public int getInputDims() {
        return X[0].length;
    }

    public int getOutDims() {
        return Y[0].length;
    }

    public void shuffle() {
        Random rnd = new Random();
        for (int i = this.getSize() - 1; i > 0; i--)
        {
            int index = rnd.nextInt(i + 1);
            // Simple swap
            double[] tmp = X[i];
            X[i] = X[index];
            X[index] = tmp;

            tmp = Y[i];
            Y[i] = Y[index];
            Y[index] = tmp;
        }
        this.currIndex = 0;
    }

    public Pair<DoubleMatrix> getNextMiniBatch(int batchsize) {
        int start = this.currIndex;
        int end = Math.min(start + batchsize, this.getSize());
        this.currIndex = end;

        if (this.currIndex >= this.getSize()) {
            this.currIndex = 0;
            return null;
        }

        double[][] bX = new double[end - start][getInputDims()];
        double[][] bY = new double[end - start][getInputDims()];
        for (int i = start; i < end; i++) {
            bX[i - start] = X[i];
            bY[i - start] = Y[i];
        }

        return new Pair<DoubleMatrix>(
                new DoubleMatrix(bX),
                new DoubleMatrix(bY)
        );
    }

}
