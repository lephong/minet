package minet.src.main.java.minet.layer;

import org.ejml.simple.*;

public interface Layer {

    public SimpleMatrix forward(SimpleMatrix X);

    public SimpleMatrix backward(SimpleMatrix gradY);
}
