package assn2.test;

import opt.EvaluationFunction;
import opt.OptimizationAlgorithm;
import opt.example.KnapsackEvaluationFunction;
import shared.writer.CSVWriter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class KnapsackTestRunner extends OptimizationTestRunner {
    /** The percentage of the space that the knapsack can hold. */
    private static final double KNAPSACK_PERCENTAGE_SPACE = 0.4;

    private String[] KNAPSACK_FIELDS = {
            "Items",
            "Copies",
            "MaxWeight",
            "MaxVolume",
            "KnapsackVolume"
    };

    private int numItems = 40;

    private int copiesEach = 1;

    private double maxWeight = 50;

    private double maxVolume = 50;

    private double knapsackVolume;

    private Random random = new Random();

    private KnapsackEvaluationFunction evaluationFunction;

    public static class Builder {
        private KnapsackTestRunner knapsackVaryItemsTest;

        public Builder() {
            this.knapsackVaryItemsTest = new KnapsackTestRunner();
        }

        public Builder setCopiesEach(int copiesEach) {
            knapsackVaryItemsTest.copiesEach = copiesEach;
            return this;
        }

        public Builder setNumItems(int numItems) {
            knapsackVaryItemsTest.numItems = numItems;
            return this;
        }

        public KnapsackTestRunner build() {
            knapsackVaryItemsTest.generateEvaluationFunction();
            return knapsackVaryItemsTest;
        }
    }

    private KnapsackTestRunner() {
    }

    public EvaluationFunction getEvaluationFunction() {
        return evaluationFunction;
    }

    private void generateEvaluationFunction() {
        int[] copies = new int[numItems];
        Arrays.fill(copies, copiesEach);
        double[] weights = new double[numItems];
        double[] volumes = new double[numItems];
        for (int i = 0; i < numItems; i++) {
            weights[i] = random.nextDouble() * maxWeight;
            volumes[i] = random.nextDouble() * maxVolume;
        }
        knapsackVolume = maxVolume * numItems * copiesEach * KNAPSACK_PERCENTAGE_SPACE;
        evaluationFunction = new KnapsackEvaluationFunction(weights, volumes, knapsackVolume, copies);
        System.out.println("Knapsack volume: " + knapsackVolume);
    }

    @Override
    public String[] getHeaders() {
        ArrayList<String> headers = new ArrayList<String>(Arrays.asList(super.getHeaders()));
        headers.addAll(Arrays.asList(KNAPSACK_FIELDS));
        return headers.toArray(new String[headers.size()]);
    }

    @Override
    protected void write(CSVWriter writer, OptimizationAlgorithm a) throws IOException {
        super.write(writer, a);
        writer.write(Integer.toString(numItems));
        writer.write(Integer.toString(copiesEach));
        writer.write(Double.toString(maxWeight));
        writer.write(Double.toString(maxVolume));
        writer.write(Double.toString(knapsackVolume));
    }
}
