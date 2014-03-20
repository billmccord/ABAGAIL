package assn2.test;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;
import opt.*;
import opt.ga.*;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.writer.CSVWriter;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class CountOnesConvergenceIterations {

    public static void main(String[] args) {
        CountOnesTestRunner testRunner = new CountOnesTestRunner.Builder().build();
        CSVWriter writer = new CSVWriter("countOnesConvergenceIterations.csv", testRunner.getHeaders());

        try {
            writer.open();
        } catch (IOException e) {
            e.printStackTrace(System.out);
        }

        runForN(writer, 50);
        runForN(writer, 100);
        runForN(writer, 500);
        runForN(writer, 1000);
        runForN(writer, 1500);

        try {
            writer.close();
        } catch (IOException e) {
            e.printStackTrace(System.out);
        }
    }

    private static void runForN(CSVWriter writer, int N) {
        CountOnesTestRunner testRunner = new CountOnesTestRunner.Builder()
                .setN(N)
                .build();

        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = testRunner.getEvaluationFunction();

        ArrayList<OptimizationAlgorithm> algorithms = new ArrayList<OptimizationAlgorithm>();

        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        algorithms.add(new RandomizedHillClimbing(hcp));
        algorithms.add(new SimulatedAnnealing(100, .95, hcp));

        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        algorithms.add(new StandardGeneticAlgorithm(200, 150, 25, gap));

        Distribution df = new DiscreteDependencyTree(.1, ranges);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        algorithms.add(new MIMIC(200, 100, pop));

        trainTestRunnerAndWrite(writer, testRunner, algorithms, 200000, 100);
    }

    private static void trainTestRunnerAndWrite(CSVWriter writer, OptimizationTestRunner testRunner,
                                                List<OptimizationAlgorithm> algorithms, int numIterations,
                                                int confirmationIterations) {
        testRunner.putConvergenceTrainerAlgorithms(algorithms, 0.0, numIterations, confirmationIterations);
        testRunner.train();
        try {
            testRunner.write(writer);
        } catch (IOException e) {
            e.printStackTrace(System.out);
        }
    }
}
