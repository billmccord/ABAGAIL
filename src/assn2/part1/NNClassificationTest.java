package assn2.part1;

import assn2.util.DataSetUtil;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import shared.ConvergenceTrainer;
import shared.DataSet;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.writer.CSVWriter;

import java.io.IOException;
import java.text.DecimalFormat;

public class NNClassificationTest {
    private static DataSet trainingSet = DataSetUtil.readNurseryTrainingDataSet();
    private static DataSet testSet = DataSetUtil.readNurseryTestDataSet();
    private static Instance[] testInstances = testSet.getInstances();

    private static final int outputLayer = 1;
    private static int inputLayer = trainingSet.getDescription().getAttributeCount() - 1;
    private static int hiddenLayer = Math.round((outputLayer + inputLayer) * 2/3);
    private static BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();

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
        BackPropagationNetwork network = factory.createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});

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
            for(Instance testInstance : testInstances) {
                network.setInputValues(testInstance.getData());
                network.run();

                predicted = Double.parseDouble(testInstance.getLabel().toString());
                actual = Double.parseDouble(network.getOutputValues().toString());

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
}
