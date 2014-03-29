package assn2.part2;

import assn2.util.AttributeLabeledDataSet;
import assn2.util.DataSetUtil;
import dist.MultivariateGaussian;
import func.EMClusterer;
import shared.DataSet;
import shared.writer.CSVWriter;
import util.linalg.Vector;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class EMClustererDatasetTests {
    public static final String[] FIELDS = {
            "k",
            "Iterations",
            "Runtime",
            "Min Distance",
            "Instance Attr Min Var",
            "Instance Attrs Max Var",
            "Instance Attr Avg Var",
            "Cluster Attr Min Var",
            "Cluster Attrs Max Var",
            "Cluster Attr Avg Var",
            "Max Var Instance Attrs"
    };

    /**
     * The test main
     *
     * @param args ignored
     */
    public static void main(String[] args) throws Exception {
        EMClustererDatasetTests emClustererDatasetTests = new EMClustererDatasetTests();
        CSVWriter writer = new CSVWriter("EMNurseryResults.csv", FIELDS);
        writer.open();
        AttributeLabeledDataSet attributeLabeledDataSet = DataSetUtil.readNurseryAttributeLabeledTrainingDataSet();
        emClustererDatasetTests.runEMTests(writer, attributeLabeledDataSet, 1);
        writer.close();

        writer = new CSVWriter("EMLungResults.csv", FIELDS);
        writer.open();
        attributeLabeledDataSet = DataSetUtil.readLungTop101AttributeLabeledTrainingDataSet();
        emClustererDatasetTests.runEMTests(writer, attributeLabeledDataSet, 1);
        writer.close();
    }

    public void runEMTests(CSVWriter writer, AttributeLabeledDataSet attributeLabeledDataSet, int numRuns) throws IOException {
        for (int k = 1; k <= 10; k++) {
            evaluateDataSet(attributeLabeledDataSet, k, 10, writer, numRuns);
            writer.nextRecord();
            evaluateDataSet(attributeLabeledDataSet, k, 100, writer, numRuns);
            writer.nextRecord();
            evaluateDataSet(attributeLabeledDataSet, k, 500, writer, numRuns);
            writer.nextRecord();
            evaluateDataSet(attributeLabeledDataSet, k, 1000, writer, numRuns);
            writer.nextRecord();
            evaluateDataSet(attributeLabeledDataSet, k, 2000, writer, numRuns);
            writer.nextRecord();
        }
    }

    public void evaluateDataSet(AttributeLabeledDataSet attributeLabeledDataSet, int k, int iterations,
                                CSVWriter writer, int numRuns) throws IOException {
        System.out.println("\nEvaluating with k = " + k + "; iterations = " + iterations);
        DataSet set = attributeLabeledDataSet.getDataSet();

        System.out.println("DataSet Instances");
        Vector attrVariance = DataSetUtil.computeVarianceOfEachVectorPos(DataSetUtil.instancesToDataVectors(set.getInstances()));
        System.out.println("Attribute variance\n" + attrVariance.toString());

        List<Double> attrVariances = DataSetUtil.arrayToList(attrVariance.getDataCopy());
        Collections.sort(attrVariances);
        Collections.reverse(attrVariances);
        System.out.println("Sorted attribute variance\n" + attrVariances.toString());

        double maxInstanceVar = attrVariances.get(0);
        double minInstanceVar = attrVariances.get(attrVariances.size() - 1);
        double avgInstanceVar = attrVariance.sum() / attrVariance.size();
        System.out.println("Instance Var Min: " + minInstanceVar + "; Max: " + maxInstanceVar + "; Avg: " + avgInstanceVar);

        List<Integer> sortedAttrInstanceVarianceIndexes = DataSetUtil.getVectorIndexesSortedByValueDesc(attrVariance);
        System.out.println(DataSetUtil.indexesToAttributes(sortedAttrInstanceVarianceIndexes, attributeLabeledDataSet));

        EMClusterer em;
        List<Integer> sortedAttrClusterVarianceIndexes;
        double totalMinDistance = 0.0;
        double start, totalTime = 0.0, timeDivisor = Math.pow(10, 9);
        double totalMinClusterVar = 0.0, totalMaxClusterVar = 0.0, totalAvgClusterVar = 0.0;

        for (double i = 0; i < numRuns; i++) {
            start = System.nanoTime();
            em = new EMClusterer(k, 1E-6, iterations);
            em.estimate(set);
            totalTime += (System.nanoTime() - start) / timeDivisor;

            int meanCount = em.getMixture().getComponents().length;
            ArrayList<Vector> means = new ArrayList<Vector>(meanCount);
            for (int j = 0; j < meanCount; j++) {
                means.add(((MultivariateGaussian) em.getMixture().getComponents()[j]).getMean());
            }

            System.out.println("\nEM Clusters");
            Vector clusterVariance = DataSetUtil.computeVarianceOfEachVectorPos(means);
            System.out.println("Cluster variance\n" + clusterVariance.toString());

            List<Double> clusterVariances = DataSetUtil.arrayToList(clusterVariance.getDataCopy());
            Collections.sort(clusterVariances);
            Collections.reverse(clusterVariances);
            System.out.println("Sorted cluster variance\n" + clusterVariances.toString());

            totalMaxClusterVar += clusterVariances.get(0);
            totalMinClusterVar += clusterVariances.get(attrVariances.size() - 1);
            totalAvgClusterVar += clusterVariance.sum() / clusterVariance.size();
            System.out.println("Cluster Var Min: " + totalMinClusterVar / (i + 1) + "; Max: " + totalMaxClusterVar / (i + 1) + "; Avg: " + totalAvgClusterVar / (i + 1));

            sortedAttrClusterVarianceIndexes = DataSetUtil.getVectorIndexesSortedByValueDesc(clusterVariance);
            System.out.println(DataSetUtil.indexesToAttributes(sortedAttrClusterVarianceIndexes, attributeLabeledDataSet));

            totalMinDistance += DataSetUtil.minDistance(sortedAttrInstanceVarianceIndexes, sortedAttrClusterVarianceIndexes);
            System.out.println("\nMin distance between sorted indexes\n" + totalMinDistance / (i + 1));

            System.out.println(em.toString());
        }

        writer.write(Integer.toString(k));
        writer.write(Integer.toString(iterations));
        writer.write(Double.toString(totalTime / (double)numRuns));
        writer.write(Double.toString(totalMinDistance / (double)numRuns));
        writer.write(Double.toString(minInstanceVar));
        writer.write(Double.toString(maxInstanceVar));
        writer.write(Double.toString(avgInstanceVar));
        writer.write(Double.toString(totalMinClusterVar / (double)numRuns));
        writer.write(Double.toString(totalMaxClusterVar / (double)numRuns));
        writer.write(Double.toString(totalAvgClusterVar / (double)numRuns));
        writer.write("\"" + DataSetUtil.indexesToAttributes(
                sortedAttrInstanceVarianceIndexes.subList(0, Math.min(sortedAttrInstanceVarianceIndexes.size(), 10)),
                attributeLabeledDataSet) + "\"");
    }
}
