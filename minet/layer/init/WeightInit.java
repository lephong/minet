// File: WeightInit.java
// An interface for all weight init classes

package minet.layer.init;

import org.jblas.*;

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
