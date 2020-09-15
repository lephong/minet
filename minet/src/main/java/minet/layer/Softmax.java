package minet.src.main.java.minet.layer;

import org.ejml.simple.*;

public class Softmax implements Layer {

    // for backward
    SimpleMatrix X; 
    
    public Softmax() {}

    @Override
    public SimpleMatrix forward(SimpleMatrix X) {
        SimpleMatrix Y = X.copy();
        this.X = X.copy();

        for (int i = 0; i < X.numRows(); i++) {
            SimpleMatrix row = X.extractVector(true, i);
            
            // stable computation for softmax: extract the largest entry
            // maxValue = max(row)
            double maxValue = row.get(0); 
            for (int j = 1; i < row.getNumElements(); j++) {
                maxValue = Math.max(maxValue, row.get(j));
            }

            // row[i] <- exp(row[i] - maxValue)
            row = row.plus(-maxValue).elementExp();
            
            // row[i] <- row[i] / (sum_j row[j])
            double norm = row.elementSum();
            Y.setRow(i, 0, row.scale(1 / norm).getDDRM().getData());
        }
        return Y;
    }

    @Override
    public SimpleMatrix backward(SimpleMatrix mBprob) {
        return null;
    }
    
}
