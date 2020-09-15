package minet.src.main.java.minet.layer;

import org.ejml.simple.*;

public class Sigmoid implements Layer {

    // for backward
    SimpleMatrix X; 
    
    public Sigmoid() {}

    @Override
    public SimpleMatrix forward(SimpleMatrix X) {
        SimpleMatrix Y = X.copy();
        this.X = X.copy();

        // Y[i] = 1 / (1 + exp(-X[i]))
        for (int i = 0; i < Y.numRows(); i++) {
            for (int j = 0; j < Y.numCols(); j++) {
                Y.set(i, j, 1 / (1 + Math.exp(-X.get(i, j))));
            }
        }
        return Y;
    }

    @Override
    public SimpleMatrix backward(SimpleMatrix gY) {
        // gX = X * (1 - X)
        return this.X.elementMult(this.X.plus(-1).scale(-1));
    }
    
}
