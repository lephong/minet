package minet.example.mnist;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;

import minet.data.Dataset;
import minet.util.Pair;


public class MNISTDataset extends Dataset<double[], Integer> {

    int inputDims; // number of input features

    public MNISTDataset(int batchsize, boolean shuffle, Random rnd) {
        super(batchsize, shuffle, rnd);
    }


    /**
     * Get the number of MNIST input features (28*28=784)
     */
    public int getInputDims() {
        return inputDims;
    }

    /**
     * Load MNIST data from file.
     */
    @Override
    public void fromFile(String path) throws IOException {
        // Input data file: 
        //     First line: [number of samples] [xDims (784)]
        //     Each following line: [input features (a list of double values, separated by spaces)] ; [output label (an integer)]

        items = new ArrayList<Pair<double[], Integer>>();

        BufferedReader br = new BufferedReader(new FileReader(path));

        // first line
        String[] ss = br.readLine().split(" ");
        int size = Integer.valueOf(ss[0]);
        inputDims = Integer.valueOf(ss[1]);

        for (int i = 0; i < size; i++) {
            ss = br.readLine().split(" ; ");
            String[] sx = ss[0].split(" ");
            double[] xs = new double[inputDims];
            Integer y = Integer.valueOf(ss[1]);
            for (int j = 0; j < sx.length; j++) {
                xs[j] = Double.parseDouble(sx[j]);
            }
            items.add(new Pair<double[], Integer>(xs, y));
        }
        br.close();
    }    
}
