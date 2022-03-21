// File: WeightInitUniform.java
// Initialize weights uniformly

package minet.layer.init;

import org.jblas.*;

/**
 * Generate a weight matrix from a uniform distribution U(minVal, maxVal)
 *
 * @author Phong Le
 */
public class WeightInitUniform implements  WeightInit {
    double minVal, maxVal;

    public WeightInitUniform(double minVal, double maxVal) {
        this.minVal = minVal;
        this.maxVal = maxVal;
    }

    @Override
    public DoubleMatrix generate(int indims, int outdims) {
        return DoubleMatrix.rand(indims, outdims).mul(maxVal - minVal).add(minVal);
    }
}
