package assn2.part1;

import assn2.util.DataSetUtil;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import opt.ga.StandardGeneticAlgorithm;
import shared.DataSet;
import shared.ErrorMeasure;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.writer.CSVWriter;

import java.io.IOException;
import java.text.DecimalFormat;

/**
 * Implementation of randomized hill climbing, simulated annealing, and genetic algorithm to
 * find optimal weights to a neural network that is classifying a dataset.
 *
 * @author Hannah Lau
 * @version 1.0
 */
public class NNOptimizationTests {
    private static DataSet trainingSet = DataSetUtil.readNurseryTrainingDataSet();
    private static Instance[] trainingInstances = trainingSet.getInstances();
    private static DataSet testSet = DataSetUtil.readNurseryTestDataSet();
    private static Instance[] testInstances = testSet.getInstances();

    private static final int outputLayer = 1;
    private static int inputLayer = trainingSet.getDescription().getAttributeCount() - 1;
    private static int hiddenLayer = Math.round((outputLayer + inputLayer) * 2/3);
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[3];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[3];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[3];
    private static String[] oaNames = {"RHC", "SA", "GA"};
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    private static double averageError = 0;

    private static CSVWriter writer;

    private static String[] header = {
            "Algorithm",
            "Iterations",
            "Correct",
            "Incorrect",
            "PercentCorrect",
            "RunTime",
            "AverageError"
    };

    public static void main(String[] args) {
        writer = new CSVWriter("nnOptimizationDefault.csv", header);
        try {
            writer.open();
        } catch (IOException e) {
            e.printStackTrace();
        }

        runTests(5);
        runTests(50);
        runTests(100);
        runTests(500);
        runTests(1000);
        runTests(2000);
        runTests(3000);
        runTests(4000);
        runTests(5000);

        try {
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void runTests(int iterations) {
        for(int i = 0; i < oa.length; i++) {
            networks[i] = factory.createClassificationNetwork(
                    new int[] {inputLayer, hiddenLayer, outputLayer});
            nnop[i] = new NeuralNetworkOptimizationProblem(trainingSet, networks[i], measure);
        }

        oa[0] = new RandomizedHillClimbing(nnop[0]);
        oa[1] = new SimulatedAnnealing(1E11, .95, nnop[1]);
        oa[2] = new StandardGeneticAlgorithm(200, 100, 10, nnop[2]);

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime;
            int correct = 0, incorrect = 0;
            train(oa[i], networks[i], oaNames[i], iterations);
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(Instance testInstance : testInstances) {
                networks[i].setInputValues(testInstance.getData());
                networks[i].run();

                predicted = Double.parseDouble(testInstance.getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                if (Math.abs(predicted - actual) < 0.5) {
                    correct++;
                } else {
                    incorrect++;
                }
            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            double percentCorrect = (double)correct/ (double)(correct+incorrect)*100.0;

            results +=  "\nResults for " + oaNames[i] + ": \nCorrectly classified " + correct + " trainingInstances." +
                    "\nIncorrectly classified " + incorrect + " trainingInstances.\nPercent correctly classified: "
                    + df.format(percentCorrect) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

            try {
                writer.write(oa[i].getClass().getSimpleName());
                writer.write(Integer.toString(iterations));
                writer.write(Integer.toString(correct));
                writer.write(Integer.toString(incorrect));
                writer.write(Double.toString(percentCorrect));
                writer.write(Double.toString(trainingTime));
                writer.write(Double.toString(averageError));
                writer.nextRecord();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }

        System.out.println(results);
    }

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, String oaName, int iterations) {
        System.out.println("\nError results for " + oaName + "\n---------------------------");

        double totalError = 0.0;
        for(int i = 0; i < iterations; i++) {
            oa.train();

            double error = 0;
            for(Instance trainingInstance : trainingInstances) {
                network.setInputValues(trainingInstance.getData());
                network.run();

                Instance output = trainingInstance.getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }
            totalError += error;
            averageError = totalError / i;

            System.out.println("Training iteration: " + i + " error: " + df.format(error) + " average error: " + df.format(averageError));
        }
    }
}
