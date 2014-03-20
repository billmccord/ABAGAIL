package assn2.test;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import opt.OptimizationAlgorithm;
import opt.SimulatedAnnealing;
import opt.example.NeuralNetworkOptimizationProblem;
import shared.*;
import shared.filt.LabelSplitFilter;
import shared.reader.ArffDataSetReader;
import shared.writer.CSVWriter;

import java.io.IOException;
import java.text.DecimalFormat;

public class NNSAOptimizationTests {
    private static final String DIR_PREFIX = "src/assn2/data/";

    private static final String DATA_FILE_TEMPLATE = DIR_PREFIX + "%s/nursery-%s.arff";

    private static final String TRAINING_FILE = String.format(DATA_FILE_TEMPLATE, "training", "training");

    private static final String TEST_FILE = String.format(DATA_FILE_TEMPLATE, "test", "test");

    private static final double[] temperature = {
            1E9, 1E9, 1E9, 1E11, 1E11, 1E11, 1E13, 1E13, 1E13
    };

    private static final double[] cooling = {
            0.93, 0.95, 0.97, 0.93, 0.95, 0.97, 0.93, 0.95, 0.97
    };

    private static DataSet trainingSet = readTrainingDataSet();
    private static Instance[] trainingInstances = trainingSet.getInstances();
    private static DataSet testSet = readTestDataSet();
    private static Instance[] testInstances = testSet.getInstances();

    private static final int outputLayer = 1;
    private static int inputLayer = trainingSet.getDescription().getAttributeCount() - 1;
    private static int hiddenLayer = Math.round((outputLayer + inputLayer) * 2/3);
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
    
    private static ErrorMeasure measure = new SumOfSquaresError();

    private static BackPropagationNetwork networks[] = new BackPropagationNetwork[9];
    private static NeuralNetworkOptimizationProblem[] nnop = new NeuralNetworkOptimizationProblem[9];

    private static OptimizationAlgorithm[] oa = new OptimizationAlgorithm[9];
    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    private static double averageError = 0;

    private static CSVWriter writer;

    private static String[] header = {
            "Algorithm",
            "Temperature",
            "Cooling",
            "Iterations",
            "Correct",
            "Incorrect",
            "PercentCorrect",
            "RunTime",
            "AverageError"
    };

    public static void main(String[] args) {
        writer = new CSVWriter("nnSAOptimization.csv", header);
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
            oa[i] = new SimulatedAnnealing(temperature[i], cooling[i], nnop[i]);
        }

        for(int i = 0; i < oa.length; i++) {
            double start = System.nanoTime(), end, trainingTime, testingTime;
            int correct = 0, incorrect = 0;
            System.out.println("\nError results for SA with temp " + temperature[i]
                    + " and cooling " + cooling[i] + " \n---------------------------");
            train(oa[i], networks[i], iterations);
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            Instance optimalInstance = oa[i].getOptimal();
            networks[i].setWeights(optimalInstance.getData());

            double predicted, actual;
            start = System.nanoTime();
            for(int j = 0; j < testInstances.length; j++) {
                networks[i].setInputValues(testInstances[j].getData());
                networks[i].run();

                predicted = Double.parseDouble(testInstances[j].getLabel().toString());
                actual = Double.parseDouble(networks[i].getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            double percentCorrect = (double)correct/ (double)(correct+incorrect)*100.0;

            results +=  "\nResults for SA with temp " + temperature[i] + " and cooling " + cooling[i]
                    + ": \nCorrectly classified " + correct + " trainingInstances." +
                    "\nIncorrectly classified " + incorrect + " trainingInstances.\nPercent correctly classified: "
                    + df.format(percentCorrect) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

            try {
                writer.write(oa[i].getClass().getSimpleName());
                writer.write(Double.toString(temperature[i]));
                writer.write(Double.toString(cooling[i]));
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

    private static void train(OptimizationAlgorithm oa, BackPropagationNetwork network, int iterations) {
        double totalError = 0.0;
        for(int i = 0; i < iterations; i++) {
            oa.train();

            double error = 0;
            for(int j = 0; j < trainingInstances.length; j++) {
                network.setInputValues(trainingInstances[j].getData());
                network.run();

                Instance output = trainingInstances[j].getLabel(), example = new Instance(network.getOutputValues());
                example.setLabel(new Instance(Double.parseDouble(network.getOutputValues().toString())));
                error += measure.value(output, example);
            }
            totalError += error;
            averageError = totalError / i;

            System.out.println("Training iteration: " + i + " error: " + df.format(error) + " average error: " + df.format(averageError));
        }
    }

    private static DataSet readTrainingDataSet() {
        return readDataSet(TRAINING_FILE);
    }

    private static DataSet readTestDataSet() {
        return readDataSet(TEST_FILE);
    }

    private static DataSet readDataSet(String fileName) {
        ArffDataSetReader reader = new ArffDataSetReader(fileName);
        DataSet ds;
        try {
            ds = reader.read();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }

        LabelSplitFilter lsf = new LabelSplitFilter();
        lsf.filter(ds);

        // Try to classify if the person was recommended or not.
        for(Instance instance : ds) {
            instance.setLabel(new Instance(instance.getLabel().getDiscrete() == 0 ? 0 : 1));
        }

        System.out.println(ds);
        System.out.println(new DataSetDescription(ds));

        return ds;
    }
}
