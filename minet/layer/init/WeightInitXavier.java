// File: WeightInitXavier.java
// Initialize weights using Xavier method

package minet.layer.init;

import org.jblas.*;

/**
 * Generate a weight matrix using Xavier's method (
 * see <a href="http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf">paper, equation 16</a>).
 *
 * @author Phong Le
 */
public class WeightInitXavier implements  WeightInit {

    public WeightInitXavier() { }

    @Override
    public DoubleMatrix generate(int indims, int outdims) {
        double a = (double) (Math.sqrt(6) / Math.sqrt(indims + outdims));
        return DoubleMatrix.rand(indims, outdims).mul(2 * a).add(-a);
    }
}

