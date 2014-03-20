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

public class KnapsackConvergenceIterations {

    public static void main(String[] args) {
        KnapsackTestRunner testRunner = new KnapsackTestRunner.Builder().build();
        CSVWriter writer = new CSVWriter("knapsackConvergenceIterations.csv", testRunner.getHeaders());

        try {
            writer.open();
        } catch (IOException e) {
            e.printStackTrace(System.out);
        }

        runForItems(writer, 20);
        runForItems(writer, 40);
        runForItems(writer, 60);
        runForItems(writer, 80);
        runForItems(writer, 100);
        runForItems(writer, 200);
        runForItems(writer, 400);
        runForItems(writer, 600);

        try {
            writer.close();
        } catch (IOException e) {
            e.printStackTrace(System.out);
        }
    }

    private static void runForItems(CSVWriter writer, int numItems) {
        int copiesEach = 5;
        KnapsackTestRunner testRunner = new KnapsackTestRunner.Builder()
                .setNumItems(numItems)
                .setCopiesEach(copiesEach)
                .build();

        int[] ranges = new int[numItems];
        Arrays.fill(ranges, copiesEach + 1);
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
