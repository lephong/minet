// File: Linear.java
// Linear layer
package minet.layer;

import org.jblas.*;

import java.util.List;

/**
 * This is a class for linear layers
 *
 * @author Phong Le
 */

public class Linear implements Layer {

    DoubleMatrix W;  // weight matrix
    DoubleMatrix b;  // bias vector

    // for backward
    DoubleMatrix X;   // store input X for computing backward
    DoubleMatrix gW;  // gradient of W
    DoubleMatrix gb;  // gradient of b

    /**
     * This is an interface for initialising weights
     *
     * @author Phong Le
     */

    public interface WeightInit {
        /**
         * generate a weight matrix
         * @param indims is the number of rows
         * @param outdims is the number of columns
         * @return a weight matrix
         */
        public DoubleMatrix generate(int indims, int outdims);
    }

    public static class WeightInitUniform implements  WeightInit {
        double minVal, maxVal;

        public WeightInitUniform(double minVal, double maxVal) {
            this.minVal = minVal;
            this.maxVal = maxVal;
        }

        @Override
        public DoubleMatrix generate(int indims, int outdims) {
            return DoubleMatrix.rand(indims, outdims).mul(this.maxVal - this.minVal).add(this.minVal);
        }
    }

    public static class WeightInitNorm implements  WeightInit {
        double mean, std;

        public WeightInitNorm(double mean, double std) {
            this.mean = mean;
            this.std = std;
        }

        @Override
        public DoubleMatrix generate(int indims, int outdims) {
            return DoubleMatrix.randn(indims, outdims).mul(this.std).add(this.mean);
        }
    }

    public static class WeightInitXavier implements  WeightInit {

        public WeightInitXavier() { }

        @Override
        public DoubleMatrix generate(int indims, int outdims) {
            double a = (double) (Math.sqrt(6) / Math.sqrt(indims + outdims));
            return DoubleMatrix.rand(indims, outdims).mul(2 * a).add(-a);
        }
    }


    public Linear(int indims, int outdims, WeightInit wInit) {
        this.W = wInit.generate(indims, outdims);
        this.b = DoubleMatrix.zeros(outdims);
        this.gW = DoubleMatrix.zeros(indims, outdims);
        this.gb = DoubleMatrix.zeros(outdims);
    }

    public Linear(DoubleMatrix W, DoubleMatrix b) {
        this.W = W;
        this.b = b;
    }

    public Layer clone() {
        return new Linear(this.W.dup(), this.b.dup());
    }

    @Override
    public DoubleMatrix forward(DoubleMatrix X) {
        // Y = X * W + b
        DoubleMatrix Y = X.mmul(this.W).addiRowVector(this.b);
        this.X = X.dup();
        return Y;
    }

    @Override
    public DoubleMatrix backward(DoubleMatrix gY) {
        // gW = gX^T * gY 
        this.gW.addi(this.X.transpose().mmul(gY));

        // gb = sum_row gY
        this.gb.addi(gY.columnSums());

        // gX = gY * W^T
        return gY.mmul(this.W.transpose());
    }

    @Override
    public List<DoubleMatrix> getAllWeights(List<DoubleMatrix> weights) {
        weights.add(this.W);
        weights.add(this.b);
        return weights;
    }

    @Override
    public List<DoubleMatrix> getAllGradients(List<DoubleMatrix> grads) {
        grads.add(this.gW);
        grads.add(this.gb);
        return grads;
    }

    @Override
    public String toString() {
        return String.format("Linear: %d in, %d out", this.W.rows, this.W.columns);
    }
}
