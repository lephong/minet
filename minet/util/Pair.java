package minet.util;

public class Pair<Tx, Ty> {
    public Tx first;
    public Ty second;

    public Pair (Tx x, Ty y) {
        this.first = x;
        this.second = y;
    }
}
