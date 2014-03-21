package assn2.part1;

import assn2.util.CsvWriteable;
import assn2.util.SimpleStopWatch;
import opt.EvaluationFunction;
import opt.OptimizationAlgorithm;
import shared.ConvergenceTrainer;
import shared.FixedIterationTrainer;
import shared.IterationTrainer;
import shared.writer.CSVWriter;

import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public abstract class OptimizationTestRunner implements CsvWriteable {

    private static final String[] FIELDS = {
            "Algorithm",
            "Iterations",
            "Runtime",
            "Score"
    };

    private SimpleStopWatch stopwatch = new SimpleStopWatch();

    private Map<OptimizationAlgorithm, IterationTrainer> algorithmTrainers
            = new HashMap<OptimizationAlgorithm, IterationTrainer>();

    private Map<OptimizationAlgorithm, Double> runTimes = new HashMap<OptimizationAlgorithm, Double>();

    protected abstract EvaluationFunction getEvaluationFunction();

    public final void putFixedIterationTrainerAlgorithms(List<OptimizationAlgorithm> algorithms, int maxIterations) {
        for (OptimizationAlgorithm a : algorithms) {
            putFixedIterationTrainerAlgorithm(a, maxIterations);
        }
    }

    public final void putFixedIterationTrainerAlgorithm(OptimizationAlgorithm a, int maxIterations) {
        algorithmTrainers.put(a, new FixedIterationTrainer(a, maxIterations));
        runTimes.put(a, 0.0);
    }

    public final void putConvergenceTrainerAlgorithms(List<OptimizationAlgorithm> algorithms,
                                                      double convergenceThreshold, int maxIterations,
                                                      int confirmationIterations) {
        for (OptimizationAlgorithm a : algorithms) {
            putConvergenceTrainerAlgorithm(a, convergenceThreshold, maxIterations, confirmationIterations);
        }
    }

    public final void putConvergenceTrainerAlgorithm(OptimizationAlgorithm a, double convergenceThreshold,
                                                     int maxIterations, int confirmationIterations) {
        algorithmTrainers.put(a, new ConvergenceTrainer(a, convergenceThreshold, maxIterations, confirmationIterations));
        runTimes.put(a, 0.0);
    }

    public final void train() {
        for (Map.Entry<OptimizationAlgorithm, IterationTrainer> algorithmTrainer : algorithmTrainers.entrySet()) {
            OptimizationAlgorithm a = algorithmTrainer.getKey();
            IterationTrainer t = algorithmTrainer.getValue();
            stopwatch.start();
            t.train();
            runTimes.put(a, stopwatch.stop());
            System.out.println("Trained " + a.getClass().getSimpleName()
                    + " for " + t.getIterations() + " iterations"
                    + " in " + runTimes.get(a) + "s with optimal result: "
                    + getEvaluationFunction().value(a.getOptimal()));
        }
    }

    @Override
    public String[] getHeaders() {
        return FIELDS;
    }

    @Override
    public final void write(CSVWriter writer) throws IOException {
        for (Map.Entry<OptimizationAlgorithm, IterationTrainer> algorithmTrainer : algorithmTrainers.entrySet()) {
            write(writer, algorithmTrainer.getKey());
            writer.nextRecord();
        }
    }

    protected void write(CSVWriter writer, OptimizationAlgorithm a) throws IOException {
        writer.write(a.getClass().getSimpleName());
        writer.write(Integer.toString(algorithmTrainers.get(a).getIterations()));
        writer.write(Double.toString(runTimes.get(a)));
        writer.write(Double.toString(getEvaluationFunction().value(a.getOptimal())));
    }
}
