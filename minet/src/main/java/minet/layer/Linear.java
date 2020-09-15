package minet.src.main.java.minet.layer;

import org.ejml.simple.*;

public class Linear implements Layer {

    // W and b
    SimpleMatrix W;  // indims x outdims
    SimpleMatrix b;     // outdims

    // for backward
    SimpleMatrix X;
    SimpleMatrix gW;
    SimpleMatrix gb;

    public Linear(int indims, int outdims) {
        this.W = null;   // TODO: init with Xavier's method
        this.b = null;   // TODO: set 0
    }

    @Override
    public SimpleMatrix forward(SimpleMatrix X) {
        // Y = X * W + b
        SimpleMatrix Y = X.mult(this.W);
        for (int i = 0; i < Y.numCols(); i++) {
            Y.setRow(i, 0, Y.extractVector(true, i).plus(this.b).getDDRM().getData());
        }
        this.X = X.copy();
        return Y;
    }

    @Override
    public SimpleMatrix backward(SimpleMatrix gY) {
        // gW = gX^T * gY 
        this.gW = this.X.transpose().mult(gY);

        // gb = sum_row gY
        for (int i = 0; i < gY.numRows(); i++) {
            this.gb = this.gb.plus(gY.extractVector(true, i));
        }

        // gX = gY * W^T
        return gY.mult(this.W.transpose());
    }
    
}
