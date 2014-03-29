package assn2.part2;

import assn2.util.AttributeLabeledDataSet;
import assn2.util.DataSetUtil;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import shared.ConvergenceTrainer;
import shared.DataSet;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.filt.PrincipalComponentAnalysis;
import shared.writer.CSVWriter;

import java.io.IOException;
import java.text.DecimalFormat;

public class NNDatasetTests {

    private static String[] FIELDS = {
            "Attrs Removed"
    };

    private static String[] K_MEANS_FIELDS;
    private static String[] EM_FIELDS;

    static {
        K_MEANS_FIELDS = concatArrays(FIELDS, KMeansClustererDatasetTests.FIELDS);
        EM_FIELDS = concatArrays(FIELDS, EMClustererDatasetTests.FIELDS);
    }

    public static String[] concatArrays(String[] array1, String[] array2) {
        String[] newArray = new String[array1.length + array2.length];
        System.arraycopy(array1, 0, newArray, 0, array1.length);
        System.arraycopy(array2, 0, newArray, array1.length, array2.length);
        return newArray;
    }
    
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) throws IOException {
        NNDatasetTests tests = new NNDatasetTests();
        tests.runkMeansNurseryTests();
        tests.runkMeansLungTests();
        tests.runEMNurseryTests();
        tests.runEMLungTests();
    }

    public void runkMeansNurseryTests() throws IOException {
        CSVWriter writer = new CSVWriter("kMeansPCAFilterNurseryResults.csv", K_MEANS_FIELDS);
        writer.open();
        for (int i = 0; i < 5; i++) {
            AttributeLabeledDataSet attributeLabeledDataSet = DataSetUtil.readNurseryAttributeLabeledTrainingDataSet();
            runkMeansTest(writer, attributeLabeledDataSet, i);
        }
        writer.close();
    }

    public void runkMeansLungTests() throws IOException {
        CSVWriter writer = new CSVWriter("kMeansPCAFilterLungResults.csv", K_MEANS_FIELDS);
        writer.open();
        for (int i = 0; i < 100; i += 20) {
            AttributeLabeledDataSet attributeLabeledDataSet = DataSetUtil.readLungTop101AttributeLabeledTrainingDataSet();
            runkMeansTest(writer, attributeLabeledDataSet, i);
        }
        writer.close();
    }

    public void runEMNurseryTests() throws IOException {
        CSVWriter writer = new CSVWriter("EMPCAFilterNurseryResults.csv", EM_FIELDS);
        writer.open();
        for (int i = 0; i < 5; i++) {
            AttributeLabeledDataSet attributeLabeledDataSet = DataSetUtil.readNurseryAttributeLabeledTrainingDataSet();
            runEMTest(writer, attributeLabeledDataSet, i);
        }
    }

    public void runEMLungTests() throws IOException {
        CSVWriter writer = new CSVWriter("EMPCAFilterLungResults.csv", EM_FIELDS);
        writer.open();
        for (int i = 0; i < 100; i += 20) {
            AttributeLabeledDataSet attributeLabeledDataSet = DataSetUtil.readLungTop101AttributeLabeledTrainingDataSet();
            runEMTest(writer, attributeLabeledDataSet, i);
        }
        writer.close();
    }

    public void runkMeansTest(CSVWriter writer, AttributeLabeledDataSet attributeLabeledDataSet, int removeAttrs)
            throws IOException {
        filterSet(attributeLabeledDataSet, removeAttrs);
        KMeansClustererDatasetTests kMeansClustererDatasetTests = new KMeansClustererDatasetTests();
        for (int k = 1; k <= 20; k++) {
            writer.write(Integer.toString(removeAttrs));
            kMeansClustererDatasetTests.evaluateDataSet(attributeLabeledDataSet, k, writer, 10);
            writer.nextRecord();
        }
    }

    public void runEMTest(CSVWriter writer, AttributeLabeledDataSet attributeLabeledDataSet, int removeAttrs)
            throws IOException {
        filterSet(attributeLabeledDataSet, removeAttrs);
        EMClustererDatasetTests emClustererDatasetTests = new EMClustererDatasetTests();
        for (int k = 1; k <= 10; k++) {
            writer.write(Integer.toString(removeAttrs));
            emClustererDatasetTests.evaluateDataSet(attributeLabeledDataSet, k, 1000, writer, 1);
            writer.nextRecord();
        }
    }

    public void filterSet(AttributeLabeledDataSet attributeLabeledDataSet, int removeAttrs) {
        double start = System.nanoTime();
        DataSet set = attributeLabeledDataSet.getDataSet();
        PrincipalComponentAnalysis filter = new PrincipalComponentAnalysis(set,
                set.getInstances()[0].size() - removeAttrs);
        filter.filter(set);
        System.out.println("Time to PCA with removal of " + removeAttrs + " attrs: "
                + (System.nanoTime() - start) / Math.pow(10, 9));
    }

    private void runNNTrainer(CSVWriter writer, DataSet trainingSet, DataSet testSet) {
        int outputLayer = 1;
        int inputLayer = trainingSet.getDescription().getAttributeCount() - 1;
        int hiddenLayer = Math.round((outputLayer + inputLayer) * 2/3);
        BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
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
        for(Instance testInstance : testSet.getInstances()) {
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

        DecimalFormat df = new DecimalFormat("0.000");
        String results = "\nResults: \nCorrectly classified " + correct + " trainingInstances." +
                "\nIncorrectly classified " + incorrect + " trainingInstances.\nPercent correctly classified: "
                + df.format(percentCorrect) + "%\nTraining time: " + df.format(trainingTime)
                + " seconds\nTesting time: " + df.format(testingTime) + " seconds\n";
        System.out.println(results);
    }
}
