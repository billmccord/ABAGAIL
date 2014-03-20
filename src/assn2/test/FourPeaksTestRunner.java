package assn2.test;

import opt.EvaluationFunction;
import opt.OptimizationAlgorithm;
import opt.example.FourPeaksEvaluationFunction;
import shared.writer.CSVWriter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class FourPeaksTestRunner extends OptimizationTestRunner {

    private String[] FOUR_PEAKS_FIELDS = {
            "N",
            "T"
    };

    private int n = 100;

    private int t = 10;

    private FourPeaksEvaluationFunction evaluationFunction;

    public static class Builder {
        private FourPeaksTestRunner knapsackVaryItemsTest;

        public Builder() {
            this.knapsackVaryItemsTest = new FourPeaksTestRunner();
        }

        public Builder setN(int n) {
            knapsackVaryItemsTest.n = n;
            return this;
        }

        public Builder setT(int t) {
            knapsackVaryItemsTest.t = t;
            return this;
        }

        public FourPeaksTestRunner build() {
            knapsackVaryItemsTest.generateEvaluationFunction();
            return knapsackVaryItemsTest;
        }
    }

    private FourPeaksTestRunner() {
    }

    public EvaluationFunction getEvaluationFunction() {
        return evaluationFunction;
    }

    private void generateEvaluationFunction() {
        evaluationFunction = new FourPeaksEvaluationFunction(t);
    }

    @Override
    public String[] getHeaders() {
        ArrayList<String> headers = new ArrayList<String>(Arrays.asList(super.getHeaders()));
        headers.addAll(Arrays.asList(FOUR_PEAKS_FIELDS));
        return headers.toArray(new String[headers.size()]);
    }

    @Override
    protected void write(CSVWriter writer, OptimizationAlgorithm a) throws IOException {
        super.write(writer, a);
        writer.write(Integer.toString(n));
        writer.write(Integer.toString(t));
    }
}
