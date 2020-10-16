// File: Linear.java
// Linear layer
package minet.layer;

import org.jblas.*;

import java.util.List;

/**
 * A class for linear layers (Y = XW + b)
 *
 * @author Phong Le
 */
public class Linear implements Layer, java.io.Serializable {			

	private static final long serialVersionUID = -10435336293457306L;
	
	DoubleMatrix W;  // weight matrix
    DoubleMatrix b;  // bias vector

    // for backward
    DoubleMatrix X;   // store input X for computing backward
    DoubleMatrix gW;  // gradient of W
    DoubleMatrix gb;  // gradient of b

    /**
     * An interface for weight initialization.
     *
     * @author Phong Le
     */
    public interface WeightInit {
        /**
         * Generate a weight matrix.
         * @param indims the number of rows
         * @param outdims the number of columns
         * @return an [indims x outdims] matrix
         */
        public DoubleMatrix generate(int indims, int outdims);
    }

    /**
     * Generate a weight matrix from a uniform distribution U(minVal, maxVal)
     *
     * @author Phong Le
     */
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

    /**
     * Generate a weight matrix from a normal distribution N(mean, std).
     *
     * @author Phong Le
     */
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

    /**
     * Generate a weight matrix using Xavier's method (
     * see <a href="http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf">paper, equation 16</a>).
     *
     * @author Phong Le
     */
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

    @Override
    public DoubleMatrix forward(DoubleMatrix X) {
        // Y = X * W + b
        DoubleMatrix Y = X.mmul(this.W).addiRowVector(this.b);
        this.X = X.dup();
        return Y;
    }

    @Override
    public DoubleMatrix backward(DoubleMatrix gY) {
        // gW = X^T * gY
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
