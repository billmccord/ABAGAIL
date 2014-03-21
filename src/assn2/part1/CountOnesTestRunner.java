package assn2.part1;

import opt.EvaluationFunction;
import opt.OptimizationAlgorithm;
import opt.example.CountOnesEvaluationFunction;
import shared.writer.CSVWriter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;

public class CountOnesTestRunner extends OptimizationTestRunner {

    private String[] COUNT_ONES_FIELDS = {
            "N"
    };

    private int n = 100;

    private CountOnesEvaluationFunction evaluationFunction;

    public static class Builder {
        private CountOnesTestRunner countOnesTestRunner;

        public Builder() {
            this.countOnesTestRunner = new CountOnesTestRunner();
        }

        public Builder setN(int n) {
            countOnesTestRunner.n = n;
            return this;
        }

        public CountOnesTestRunner build() {
            countOnesTestRunner.generateEvaluationFunction();
            return countOnesTestRunner;
        }
    }

    private CountOnesTestRunner() {
    }

    public EvaluationFunction getEvaluationFunction() {
        return evaluationFunction;
    }

    private void generateEvaluationFunction() {
        evaluationFunction = new CountOnesEvaluationFunction();
    }

    @Override
    public String[] getHeaders() {
        ArrayList<String> headers = new ArrayList<String>(Arrays.asList(super.getHeaders()));
        headers.addAll(Arrays.asList(COUNT_ONES_FIELDS));
        return headers.toArray(new String[headers.size()]);
    }

    @Override
    protected void write(CSVWriter writer, OptimizationAlgorithm a) throws IOException {
        super.write(writer, a);
        writer.write(Integer.toString(n));
    }
}
