package assn2.test;

import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import shared.*;
import shared.filt.LabelSplitFilter;
import shared.reader.ArffDataSetReader;
import shared.writer.CSVWriter;

import java.io.IOException;
import java.text.DecimalFormat;

public class NNClassificationTest {
    private static final String DIR_PREFIX = "src/assn2/data/";

    private static final String DATA_FILE_TEMPLATE = DIR_PREFIX + "%s/nursery-%s.arff";

    private static final String TRAINING_FILE = String.format(DATA_FILE_TEMPLATE, "training", "training");

    private static final String TEST_FILE = String.format(DATA_FILE_TEMPLATE, "test", "test");

    private static DataSet trainingSet = readTrainingDataSet();
    private static DataSet testSet = readTestDataSet();
    private static Instance[] testInstances = testSet.getInstances();

    private static final int outputLayer = 1;
    private static int inputLayer = trainingSet.getDescription().getAttributeCount() - 1;
    private static int hiddenLayer = Math.round((outputLayer + inputLayer) * 2/3);
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

    private static BackPropagationNetwork network;

    private static String results = "";

    private static DecimalFormat df = new DecimalFormat("0.000");

    private static CSVWriter writer;

    private static String[] header = {
            "Algorithm",
            "Iterations",
            "Correct",
            "Incorrect",
            "PercentCorrect",
            "RunTime"
    };

    public static void main(String[] args) {
        writer = new CSVWriter("nnOptimizationDefault.csv", header);
        try {
            writer.open();
        } catch (IOException e) {
            e.printStackTrace();
        }

        runTests();
        try {
            writer.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void runTests() {
        network = factory.createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});

            double start = System.nanoTime(), end, trainingTime, testingTime;
            int correct = 0, incorrect = 0;
            ConvergenceTrainer trainer = new ConvergenceTrainer(
                    new BatchBackPropagationTrainer(trainingSet, network,
                            new SumOfSquaresError(), new RPROPUpdateRule()));
            trainer.train();
            System.out.println("Convergence in " + trainer.getIterations() + " iterations");
            end = System.nanoTime();
            trainingTime = end - start;
            trainingTime /= Math.pow(10,9);

            double predicted, actual;
            start = System.nanoTime();
            for(int i = 0; i < testInstances.length; i++) {
                network.setInputValues(testInstances[i].getData());
                network.run();

                predicted = Double.parseDouble(testInstances[i].getLabel().toString());
                actual = Double.parseDouble(network.getOutputValues().toString());

                double trash = Math.abs(predicted - actual) < 0.5 ? correct++ : incorrect++;
            }
            end = System.nanoTime();
            testingTime = end - start;
            testingTime /= Math.pow(10,9);

            double percentCorrect = (double)correct/ (double)(correct+incorrect)*100.0;

            results +=  "\nResults for BackPropagation" + ": \nCorrectly classified " + correct + " trainingInstances." +
                    "\nIncorrectly classified " + incorrect + " trainingInstances.\nPercent correctly classified: "
                    + df.format(percentCorrect) + "%\nTraining time: " + df.format(trainingTime)
                    + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";

            try {
                writer.write("BackPropagation");
                writer.write(Integer.toString(trainer.getIterations()));
                writer.write(Integer.toString(correct));
                writer.write(Integer.toString(incorrect));
                writer.write(Double.toString(percentCorrect));
                writer.write(Double.toString(trainingTime));
                writer.nextRecord();
            } catch (IOException e) {
                e.printStackTrace();
            }

        System.out.println(results);
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
