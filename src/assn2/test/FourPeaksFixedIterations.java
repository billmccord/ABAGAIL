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

public class FourPeaksFixedIterations {

    public static void main(String[] args) {
        int N = 100;
        int T = 100 / 10;
        FourPeaksTestRunner testRunner = new FourPeaksTestRunner.Builder()
                .setN(N)
                .setT(T)
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

        CSVWriter writer = new CSVWriter("fourPeaksFixedIterations.csv", testRunner.getHeaders());

        try {
            writer.open();
        } catch (IOException e) {
            e.printStackTrace(System.out);
        }

        trainTestRunnerAndWrite(writer, testRunner, algorithms, 1);
        trainTestRunnerAndWrite(writer, testRunner, algorithms, 50);
        trainTestRunnerAndWrite(writer, testRunner, algorithms, 100);
        trainTestRunnerAndWrite(writer, testRunner, algorithms, 500);
        trainTestRunnerAndWrite(writer, testRunner, algorithms, 1000);
        trainTestRunnerAndWrite(writer, testRunner, algorithms, 5000);
        trainTestRunnerAndWrite(writer, testRunner, algorithms, 10000);
        trainTestRunnerAndWrite(writer, testRunner, algorithms, 50000);
        trainTestRunnerAndWrite(writer, testRunner, algorithms, 100000);

        try {
            writer.close();
        } catch (IOException e) {
            e.printStackTrace(System.out);
        }
    }

    private static void trainTestRunnerAndWrite(CSVWriter writer, OptimizationTestRunner testRunner,
                                                List<OptimizationAlgorithm> algorithms, int numIterations) {
        testRunner.putFixedIterationTrainerAlgorithms(algorithms, numIterations);
        testRunner.train();
        try {
            testRunner.write(writer);
        } catch (IOException e) {
            e.printStackTrace(System.out);
        }
    }
}
