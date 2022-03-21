// File: Dataset.java
// Dataset class
package minet.data;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.io.IOException;
import java.util.Random;

import minet.util.Pair;

/**
 * Abstract class for all Dataset classes. 
 * Each child class has to specify Tin (type of input) and Tout (type of output).
 * @author Phong Le
 */
public abstract class Dataset<Tin, Tout> implements java.io.Serializable{

    protected ArrayList<Pair<Tin, Tout>> items; // list of samples in this dataset. Each item is a sample, which consists of the sample's input and its output label(s). 
    protected boolean shuffle; // if true, shuffle the dataset once a pass (i.e., an epoch) over the data is finished.
    protected int currIndex;  // index of the starting sample of the current batch.
    protected int batchsize; // batch size
    protected Random rnd; // random generator

    /**
     * Constructor for Dataset
     * @param batchsize (int) size of each mini-batch
     * @param shuffle (boolean) if true, shuffle the dataset at the beginning of each epoch
     * @param rnd (java.util.Random) random generator for the shuffling
     */
    public Dataset(int batchsize, boolean shuffle, Random rnd){
        this.batchsize = batchsize; 
        this.currIndex = 0;
        this.shuffle = shuffle;        
        this.rnd = rnd;
    }

    /**
     * Load items from file
     * @param path Path to file to load      
     */
    abstract public void fromFile(String path) throws IOException;

    /**
     * Get the number of items in the dataset.
     * @return the number of items
     */
    public int getSize() {
        return items.size();
    }

    /**
     * Must be called before using this dataset
     */
    public void reset() {
        this.currIndex = 0;
        if (this.shuffle){
            Collections.shuffle(items, rnd);
        }
    }

    /**
     * Get a minibatch of size batchsize
     * @return a list of pair of X (feature values) and Y (labels)
     */
    public List<Pair<Tin, Tout>> getNextMiniBatch() {
        // stop the epoch
    	if (currIndex >= items.size()) {
            this.reset();
            return null;
        }
    	
        // get the next minibatch
        int start = currIndex;
        int end = Math.min(start + batchsize, items.size());
        currIndex = end;
        return items.subList(start, end);
    }

}
