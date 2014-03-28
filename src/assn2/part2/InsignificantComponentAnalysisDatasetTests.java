package assn2.part2;

import assn2.util.AttributeLabeledDataSet;
import assn2.util.DataSetUtil;
import shared.DataSet;
import shared.filt.InsignificantComponentAnalysis;
import shared.writer.CSVWriter;

import java.io.IOException;

/**
 * A class for testing
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class InsignificantComponentAnalysisDatasetTests {

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
        InsignificantComponentAnalysisDatasetTests tests = new InsignificantComponentAnalysisDatasetTests();
        tests.runkMeansNurseryTests();
        tests.runkMeansLungTests();
        tests.runEMNurseryTests();
        tests.runEMLungTests();
    }

    public void runkMeansNurseryTests() throws IOException {
        CSVWriter writer = new CSVWriter("kMeansInCAFilterNurseryResults.csv", K_MEANS_FIELDS);
        writer.open();
        for (int i = 0; i < 5; i++) {
            AttributeLabeledDataSet attributeLabeledDataSet = DataSetUtil.readNurseryAttributeLabeledTrainingDataSet();
            runkMeansTest(writer, attributeLabeledDataSet, i);
        }
        writer.close();
    }

    public void runkMeansLungTests() throws IOException {
        CSVWriter writer = new CSVWriter("kMeansInCAFilterLungResults.csv", K_MEANS_FIELDS);
        writer.open();
        for (int i = 0; i < 100; i += 20) {
            AttributeLabeledDataSet attributeLabeledDataSet = DataSetUtil.readLungTop101AttributeLabeledTrainingDataSet();
            runkMeansTest(writer, attributeLabeledDataSet, i);
        }
        writer.close();
    }

    public void runEMNurseryTests() throws IOException {
        CSVWriter writer = new CSVWriter("EMInCAFilterNurseryResults.csv", EM_FIELDS);
        writer.open();
        for (int i = 0; i < 5; i++) {
            AttributeLabeledDataSet attributeLabeledDataSet = DataSetUtil.readNurseryAttributeLabeledTrainingDataSet();
            runEMTest(writer, attributeLabeledDataSet, i);
        }
    }

    public void runEMLungTests() throws IOException {
        CSVWriter writer = new CSVWriter("EMInCAFilterLungResults.csv", EM_FIELDS);
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
        DataSet set = attributeLabeledDataSet.getDataSet();
        InsignificantComponentAnalysis filter = new InsignificantComponentAnalysis(set,
                set.getInstances()[0].size() - removeAttrs);
        filter.filter(set);
    }
}
