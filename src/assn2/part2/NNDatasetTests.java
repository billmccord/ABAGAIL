package assn2.part2;

import assn2.util.DataSetUtil;
import func.EMClusterer;
import func.KMeansClusterer;
import func.nn.backprop.BackPropagationNetwork;
import func.nn.backprop.BackPropagationNetworkFactory;
import func.nn.backprop.BatchBackPropagationTrainer;
import func.nn.backprop.RPROPUpdateRule;
import shared.ConvergenceTrainer;
import shared.DataSet;
import shared.Instance;
import shared.SumOfSquaresError;
import shared.filt.*;
import shared.writer.CSVWriter;

import java.io.IOException;

public class NNDatasetTests {

    private static String[] header = {
            "DimReductionAlgo",
            "AttrsRemoved",
            "ClusteringAlgo",
            "k",
            "AvgIterations",
            "AvgCorrect",
            "AvgIncorrect",
            "AvgPercentCorrect",
            "AvgRunTime"
    };

    private static final int RAND_TRIALS = 3;
    private static final DataSet NURSERY_TRAINING_SET = DataSetUtil.readNurseryTrainingDataSet();
    private static final DataSet NURSERY_TEST_SET = DataSetUtil.readNurseryTestDataSet();
    private static final DataSet LUNG_TRAINING_SET = DataSetUtil.readLungTop101AttributeTrainingDataSet();
    private static final DataSet LUNG_TEST_SET = DataSetUtil.readLungTop101AttributeTestDataSet();

    private int k;
    private int maxK;
    private int totalIterations;
    private int totalCorrect;
    private int totalIncorrect;
    private double totalRuntime;
    private int removeAttrsIncr;
    private int maxRemoveAttrs;

    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) throws IOException {
        NNDatasetTests tests = new NNDatasetTests(1, 5, 10);
        CSVWriter writer = new CSVWriter("NNMultiNurseryResults.csv", header);
        writer.open();
        System.out.println("Running All Nursery Tests");
        tests.runAllTests(writer, NURSERY_TRAINING_SET, NURSERY_TEST_SET);
        writer.close();

        tests = new NNDatasetTests(20, 100, 7);
        writer = new CSVWriter("NNMultiLungResults.csv", header);
        writer.open();
        System.out.println("Running All Lung Tests");
        tests.runAllTests(writer, LUNG_TRAINING_SET, LUNG_TEST_SET);
        writer.close();
    }

    public NNDatasetTests(int removeAttrsIncr, int maxRemoveAttrs, int maxK) {
        this.removeAttrsIncr = removeAttrsIncr;
        this.maxRemoveAttrs = maxRemoveAttrs;
        this.maxK = maxK;
    }

    public void runAllTests(CSVWriter writer, DataSet trainingSet, DataSet testSet) throws IOException {
        for(k = 2; k <= maxK; k++) {
            System.out.println("Starting k = " + k);
            runKMeansTests(writer, trainingSet, testSet);
            runEMTests(writer, trainingSet, testSet);
            runPCAKMeansTests(writer, trainingSet, testSet);
            runPCAEMTests(writer, trainingSet, testSet);
            runICAKMeansTests(writer, trainingSet, testSet);
            runICAEMTests(writer, trainingSet, testSet);
            runInCAKMeansTests(writer, trainingSet, testSet);
            runInCAEMTests(writer, trainingSet, testSet);
            runRPKMeansTests(writer, trainingSet, testSet);
            runRPEMTests(writer, trainingSet, testSet);
        }
    }

    public void runKMeansTests(CSVWriter writer, DataSet trainingSet, DataSet testSet) throws IOException {
        System.out.println("Running kMeans Tests");
        runKMeansTest(null, trainingSet.copy(), testSet.copy());
        writeResultsAndReset(writer, "None", 0, "kMeans", 1);
    }

    public void runEMTests(CSVWriter writer, DataSet trainingSet, DataSet testSet) throws IOException {
        System.out.println("Running EM Tests");
        runEMTest(null, trainingSet.copy(), testSet.copy());
        writeResultsAndReset(writer, "None", 0, "EM", 1);
    }

    public void runPCAKMeansTests(CSVWriter writer, DataSet trainingSet, DataSet testSet) throws IOException {
        System.out.println("Running PCA kMeans Tests");
        for (int removeAttrs = 0; removeAttrs < maxRemoveAttrs; removeAttrs += removeAttrsIncr) {
            System.out.println("Remove Attrs #" + removeAttrs);
            runKMeansTest(createPCAFilter(trainingSet, removeAttrs), trainingSet.copy(), testSet.copy());
            writeResultsAndReset(writer, "PCA", removeAttrs, "kMeans", 1);
        }
    }

    public void runPCAEMTests(CSVWriter writer, DataSet trainingSet, DataSet testSet) throws IOException {
        System.out.println("Running PCA EM Tests");
        for (int removeAttrs = 0; removeAttrs < maxRemoveAttrs; removeAttrs += removeAttrsIncr) {
            System.out.println("Remove Attrs #" + removeAttrs);
            runEMTest(createPCAFilter(trainingSet, removeAttrs), trainingSet.copy(), testSet.copy());
            writeResultsAndReset(writer, "PCA", removeAttrs, "EM", 1);
        }
    }

    public ReversibleFilter createPCAFilter(DataSet set, int removeAttrs) {
        try {
            System.out.println("Creating PCA FIlter");
            return new PrincipalComponentAnalysis(set,
                    set.getInstances()[0].size() - removeAttrs);
        } catch (Exception e) {
            createPCAFilter(set, removeAttrs);
        }
        return null;
    }

    public void runICAKMeansTests(CSVWriter writer, DataSet trainingSet, DataSet testSet) throws IOException {
        System.out.println("Running ICA kMeans Tests");
        for (int removeAttrs = 0; removeAttrs < maxRemoveAttrs; removeAttrs += removeAttrsIncr) {
            System.out.println("Remove Attrs #" + removeAttrs);
            runKMeansTest(createICAFilter(trainingSet, removeAttrs), trainingSet.copy(), testSet.copy());
            writeResultsAndReset(writer, "ICA", removeAttrs, "kMeans", 1);
        }
    }

    public void runICAEMTests(CSVWriter writer, DataSet trainingSet, DataSet testSet) throws IOException {
        System.out.println("Running ICA EM Tests");
        for (int removeAttrs = 0; removeAttrs < maxRemoveAttrs; removeAttrs += removeAttrsIncr) {
            System.out.println("Remove Attrs #" + removeAttrs);
            runEMTest(createICAFilter(trainingSet, removeAttrs), trainingSet.copy(), testSet.copy());
            writeResultsAndReset(writer, "ICA", removeAttrs, "EM", 1);
        }
    }

    public ReversibleFilter createICAFilter(DataSet set, int removeAttrs) {
        try {
            System.out.println("Creating ICA FIlter");
            return new IndependentComponentAnalysis(set,
                    set.getInstances()[0].size() - removeAttrs);
        } catch (Exception e) {
            createICAFilter(set, removeAttrs);
        }
        return null;
    }

    public void runInCAKMeansTests(CSVWriter writer, DataSet trainingSet, DataSet testSet) throws IOException {
        System.out.println("Running InCA kMeans Tests");
        for (int removeAttrs = 0; removeAttrs < maxRemoveAttrs; removeAttrs += removeAttrsIncr) {
            System.out.println("Remove Attrs #" + removeAttrs);
            runKMeansTest(createInCAFilter(trainingSet, removeAttrs), trainingSet.copy(), testSet.copy());
            writeResultsAndReset(writer, "InCA", removeAttrs, "kMeans", 1);
        }
    }

    public void runInCAEMTests(CSVWriter writer, DataSet trainingSet, DataSet testSet) throws IOException {
        System.out.println("Running InCA EM Tests");
        for (int removeAttrs = 0; removeAttrs < maxRemoveAttrs; removeAttrs += removeAttrsIncr) {
            System.out.println("Remove Attrs #" + removeAttrs);
            runEMTest(createInCAFilter(trainingSet, removeAttrs), trainingSet.copy(), testSet.copy());
            writeResultsAndReset(writer, "InCA", removeAttrs, "EM", 1);
        }
    }

    public ReversibleFilter createInCAFilter(DataSet set, int removeAttrs) {
        try {
            System.out.println("Creating InCA FIlter");
            return new InsignificantComponentAnalysis(set,
                    set.getInstances()[0].size() - removeAttrs);
        } catch (Exception e) {
            createInCAFilter(set, removeAttrs);
        }
        return null;
    }

    public void runRPKMeansTests(CSVWriter writer, DataSet trainingSet, DataSet testSet) throws IOException {
        System.out.println("Running RP kMeans Tests");
        for (int removeAttrs = 0; removeAttrs < maxRemoveAttrs; removeAttrs += removeAttrsIncr) {
            System.out.println("Remove Attrs #" + removeAttrs);
            for (int i = 0; i < RAND_TRIALS; i++) {
                System.out.println("Trial #" + (i+1));
                runKMeansTest(createRPFilter(trainingSet, removeAttrs), trainingSet.copy(), testSet.copy());
            }
            writeResultsAndReset(writer, "RP", removeAttrs, "kMeans", RAND_TRIALS);
        }
    }

    public void runRPEMTests(CSVWriter writer, DataSet trainingSet, DataSet testSet) throws IOException {
        System.out.println("Running RP EM Tests");
        for (int removeAttrs = 0; removeAttrs < maxRemoveAttrs; removeAttrs += removeAttrsIncr) {
            System.out.println("Remove Attrs #" + removeAttrs);
            for (int i = 0; i < RAND_TRIALS; i++) {
                System.out.println("Trial #" + (i+1));
                runEMTest(createRPFilter(trainingSet, removeAttrs), trainingSet.copy(), testSet.copy());
            }
            writeResultsAndReset(writer, "RP", removeAttrs, "EM", RAND_TRIALS);
        }
    }

    public ReversibleFilter createRPFilter(DataSet set, int removeAttrs) {
        try {
            System.out.println("Creating RP FIlter");
            return new RandomizedProjectionFilter(
                    set.getInstances()[0].size() - removeAttrs,
                    set.getInstances()[0].size());
        } catch (Exception e) {
            createRPFilter(set, removeAttrs);
        }
        return null;
    }

    public void runKMeansTest(ReversibleFilter filter, DataSet trainingSet, DataSet testSet) {
        double start = System.nanoTime();
        if (filter != null) {
            filter.filter(trainingSet);
        }
        kMeansClusterSet(trainingSet, k);
        if (filter != null) {
            filter.filter(testSet);
        }
        kMeansClusterSet(testSet, k);
        runNNTrainer(trainingSet, testSet);
        totalRuntime += (System.nanoTime() - start);
    }

    public void kMeansClusterSet(DataSet set, int k) {
        System.out.println("kMeans Clustering");
        KMeansClusterer km = new KMeansClusterer(k);
        km.estimate(set);
        km.addClusterAsAttribute(set);
    }

    public void runEMTest(ReversibleFilter filter, DataSet trainingSet, DataSet testSet) {
        double start = System.nanoTime();
        if (filter != null) {
            filter.filter(trainingSet);
        }
        EMClusterSet(trainingSet, k);
        if (filter != null) {
            filter.filter(testSet);
        }
        EMClusterSet(testSet, k);
        runNNTrainer(trainingSet, testSet);
        totalRuntime += (System.nanoTime() - start);
    }

    public void EMClusterSet(DataSet set, int k) {
        System.out.println("EM Clustering");
        try {
            EMClusterer em = new EMClusterer(k, 1E-6, 1000);
            em.estimate(set);
            em.addClusterAsAttribute(set);
        } catch (Exception e) {
            EMClusterSet(set, k);
        }
    }

    private void runNNTrainer(DataSet trainingSet, DataSet testSet) {
        int outputLayer = 1;
        int inputLayer = trainingSet.getDescription().getAttributeCount() - 1;
        int hiddenLayer = Math.round((outputLayer + inputLayer) * 2/3);
        BackPropagationNetworkFactory factory = new BackPropagationNetworkFactory();
        BackPropagationNetwork network = factory.createClassificationNetwork(new int[] {inputLayer, hiddenLayer, outputLayer});

        ConvergenceTrainer trainer = new ConvergenceTrainer(
                new BatchBackPropagationTrainer(trainingSet, network,
                        new SumOfSquaresError(), new RPROPUpdateRule()));
        trainer.train();
        totalIterations += trainer.getIterations();

        double predicted, actual;
        for(Instance testInstance : testSet.getInstances()) {
            network.setInputValues(testInstance.getData());
            network.run();

            predicted = Double.parseDouble(testInstance.getLabel().toString());
            actual = Double.parseDouble(network.getOutputValues().toString());

            if (Math.abs(predicted - actual) < 0.5) {
                totalCorrect++;
            } else {
                totalIncorrect++;
            }
        }
    }

    private void writeResultsAndReset(CSVWriter writer, String dimReducAlgo, int attrsRemoved, String clusterAlgo,
                                      int numTrials)
            throws IOException {
        writer.write(dimReducAlgo);
        writer.write(Integer.toString(attrsRemoved));
        writer.write(clusterAlgo);
        writer.write(Integer.toString(k));
        double avgIterations = (double)totalIterations / (double)numTrials;
        writer.write(Double.toString(avgIterations));
        double avgCorrect = (double)totalCorrect / (double)numTrials;
        writer.write(Double.toString(avgCorrect));
        double avgIncorrect = (double)totalIncorrect / (double)numTrials;
        writer.write(Double.toString(avgIncorrect));
        double percentCorrect = (double)totalCorrect/ (double)(totalCorrect + totalIncorrect)*100.0;
        writer.write(Double.toString(percentCorrect));
        double avgRuntime = totalRuntime / numTrials / Math.pow(10,9);
        writer.write(Double.toString(avgRuntime));
        writer.nextRecord();

        totalIterations = 0;
        totalCorrect = 0;
        totalIncorrect = 0;
        totalRuntime = 0;

        System.out.println(dimReducAlgo + " reduction with " + attrsRemoved + " attrs removed and " + clusterAlgo
                + " k = " + k + " avgIterations = " + avgIterations + " avgPercentCorrect = " + percentCorrect
                + " avgRuntime = " + avgRuntime);
    }
}
