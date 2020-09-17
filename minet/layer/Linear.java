package minet.layer;

import minet.loss.CrossEntropy;
import org.jblas.*;

import java.util.List;

public class Linear implements Layer {

    // W and b
    public DoubleMatrix W;  // indims x outdims
    public DoubleMatrix b;  // outdims

    // for backward
    public DoubleMatrix X;
    public DoubleMatrix gW;
    public DoubleMatrix gb;

    public interface WeightInit {
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
    public List<double[]> getAllWeights(List<double[]> weights) {
        weights.add(this.W.data);
        weights.add(this.b.data);
        return weights;
    }

    @Override
    public List<double[]> getAllGradients(List<double[]> grads) {
        grads.add(this.gW.data);
        grads.add(this.gb.data);
        return grads;
    }

    @Override
    public void resetGradients() {
        this.gW.fill(0);
        this.gb.fill(0);
    }
}
