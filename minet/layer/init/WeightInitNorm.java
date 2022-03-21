// File: WeightInitNorm.java
// Initialize weights with normal distribution

package minet.layer.init;

import org.jblas.*;

/**
 * Generate a weight matrix from a normal distribution N(mean, std).
 *
 * @author Phong Le
 */
public class WeightInitNorm implements  WeightInit {
    double mean, std;

    public WeightInitNorm(double mean, double std) {
        this.mean = mean;
        this.std = std;
    }

    @Override
    public DoubleMatrix generate(int indims, int outdims) {
        return DoubleMatrix.randn(indims, outdims).mul(std).add(mean);
    }
}
