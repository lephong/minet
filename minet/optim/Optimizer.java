package minet.optim;

public interface Optimizer {

    public void resetGradients();

    public void updateWeights();
}
