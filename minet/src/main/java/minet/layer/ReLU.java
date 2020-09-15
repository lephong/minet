package minet.src.main.java.minet.layer;

import org.ejml.simple.*;

public class ReLU implements Layer {

    // for backward
    SimpleMatrix X; 
    
    public ReLU() {}

    @Override
    public SimpleMatrix forward(SimpleMatrix X) {
        SimpleMatrix Y = X.copy();
        this.X = X.copy();

        // Y[i] = max(0, X[i])
        for (int i = 0; i < Y.numRows(); i++) {
            for (int j = 0; j < Y.numCols(); j++) {
                Y.set(i, j, Math.max(0, X.get(i, j)));
            }
        }
        return Y;
    }

    @Override
    public SimpleMatrix backward(SimpleMatrix gY) {
        // gX[i] = 0 if X[i] == 0 else 1
        SimpleMatrix gX = gY.copy();
        gX.fill(0);
        for (int i = 0; i < gX.numRows(); i++) {
            for (int j = 0; j < gX.numCols(); j++) {
                if (this.X.get(i, j) > 0)
                    gX.set(i, j, 1);
            }
        }

        return gX;
    }
    
}
