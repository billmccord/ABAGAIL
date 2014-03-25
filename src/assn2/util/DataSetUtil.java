package assn2.util;

import shared.DataSet;
import shared.Instance;
import shared.filt.LabelSplitFilter;
import shared.reader.ArffDataSetReader;
import util.linalg.DenseVector;
import util.linalg.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class DataSetUtil {
    private static final String DIR_PREFIX = "src/assn2/data/";

    private static final String NURSERY_DATA_FILE_TEMPLATE = DIR_PREFIX + "%s/nursery-%s.arff";

    private static final String NURSERY_TRAINING_FILE = String.format(NURSERY_DATA_FILE_TEMPLATE, "training", "training");

    private static final String NURSERY_TEST_FILE = String.format(NURSERY_DATA_FILE_TEMPLATE, "test", "test");

    private static final String LUNG_FULL_DATA_FILE_TEMPLATE = DIR_PREFIX + "%s/lung-qn-%s.arff";

    private static final String LUNG_FULL_TRAINING_FILE = String.format(LUNG_FULL_DATA_FILE_TEMPLATE, "training", "training");

    private static final String LUNG_FULL_TEST_FILE = String.format(LUNG_FULL_DATA_FILE_TEMPLATE, "test", "test");

    private static final String LUNG_TOP_X_DATA_FILE_TEMPLATE = DIR_PREFIX + "%s/lung-qn-%s-top%d.arff";

    private static final String LUNG_TOP_101_ATTR_TRAINING_FILE = String.format(LUNG_TOP_X_DATA_FILE_TEMPLATE, "training", "training", 101);

    private static final String LUNG_TOP_101_ATTR_TEST_FILE = String.format(LUNG_TOP_X_DATA_FILE_TEMPLATE, "test", "test", 101);

    public static DataSet readNurseryTrainingDataSet() {
        return readNurseryAttributeLabeledTrainingDataSet().getDataSet();
    }

    public static AttributeLabeledDataSet readNurseryAttributeLabeledTrainingDataSet() {
        AttributeLabeledDataSet attrLabeledSet = readAttributeLabeledDataSet(NURSERY_TRAINING_FILE);
        binarizeNurseryData(attrLabeledSet.getDataSet());
        return attrLabeledSet;
    }

    public static DataSet readNurseryTestDataSet() {
        return readNurseryAttributeLabeledTestDataSet().getDataSet();
    }

    public static AttributeLabeledDataSet readNurseryAttributeLabeledTestDataSet() {
        AttributeLabeledDataSet attrLabeledSet = readAttributeLabeledDataSet(NURSERY_TEST_FILE);
        binarizeNurseryData(attrLabeledSet.getDataSet());
        return attrLabeledSet;
    }

    public static DataSet readLungFullTrainingDataSet() {
        return readLungFullAttributeLabeledTrainingDataSet().getDataSet();
    }

    public static AttributeLabeledDataSet readLungFullAttributeLabeledTrainingDataSet() {
        return readAttributeLabeledDataSet(LUNG_FULL_TRAINING_FILE);
    }

    public static DataSet readLungFullTestDataSet() {
        return readLungFullAttributeLabeledTestDataSet().getDataSet();
    }

    public static AttributeLabeledDataSet readLungFullAttributeLabeledTestDataSet() {
        return readAttributeLabeledDataSet(LUNG_FULL_TEST_FILE);
    }

    public static DataSet readLungTop101AttributeTrainingDataSet() {
        return readLungTop101AttributeLabeledTrainingDataSet().getDataSet();
    }

    public static AttributeLabeledDataSet readLungTop101AttributeLabeledTrainingDataSet() {
        return readAttributeLabeledDataSet(LUNG_TOP_101_ATTR_TRAINING_FILE);
    }

    public static DataSet readLungTop1001AttributeTestDataSet() {
        return readLungTop101AttributeLabeledTestDataSet().getDataSet();
    }

    public static AttributeLabeledDataSet readLungTop101AttributeLabeledTestDataSet() {
        return readAttributeLabeledDataSet(LUNG_TOP_101_ATTR_TEST_FILE);
    }

    public static DataSet readDataSet(String fileName) {
        return readAttributeLabeledDataSet(fileName).getDataSet();
    }

    private static AttributeLabeledDataSet readAttributeLabeledDataSet(String fileName) {
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

//        System.out.println(ds);
//        System.out.println(new DataSetDescription(ds));

        return new AttributeLabeledDataSet(reader.getAttributeNames(), ds);
    }

    private static DataSet binarizeNurseryData(DataSet ds) {
        // Try to classify if the person was recommended or not.
        for(Instance instance : ds) {
            instance.setLabel(new Instance(instance.getLabel().getDiscrete() == 0 ? 0 : 1));
        }

//        System.out.println(ds);
//        System.out.println(new DataSetDescription(ds));

        return ds;
    }

    public static List<Integer> getVectorIndexesSortedByValueDesc(Vector vector) {
        DoubleArrayIndexComparator vectorComparator = new DoubleArrayIndexComparator(vector.getDataCopy());
        Integer[] vectorIndexes = vectorComparator.createIndexArray();
        Arrays.sort(vectorIndexes, vectorComparator);
        List<Integer> sortedVectorIndexes = Arrays.asList(vectorIndexes);
        Collections.reverse(sortedVectorIndexes);
        System.out.println("Sorted vector indexes from highest to lowest\n" + sortedVectorIndexes);
        return sortedVectorIndexes;
    }

    public static List<Vector> instancesToDataVectors(Instance[] instances) {
        ArrayList<Vector> vectors = new ArrayList<Vector>(instances.length);
        for (Instance instance : instances) {
            vectors.add(instance.getData());
        }

        return vectors;
    }

    public static Vector computeVarianceOfEachVectorPos(List<Vector> vectors) {
        int length = vectors.get(0).size();
        DenseVector sumAttrs = new DenseVector(length);
        for (Vector vector : vectors) {
            sumAttrs.plusEquals(vector);
        }

        double inverseLength = 1.0/((double)vectors.size());
        Vector meanAttrs = sumAttrs.times(inverseLength);

        double diff;
        DenseVector diffSquaredAttrs = new DenseVector(length);
        DenseVector sumSquaredDiffAttrs = new DenseVector(length);
        for (Vector vector : vectors) {
            for (int i = 0; i < length; i++) {
                diff = meanAttrs.get(i) - vector.get(i);
                diffSquaredAttrs.set(i, diff * diff);
            }
            sumSquaredDiffAttrs.plusEquals(diffSquaredAttrs);
        }

        Vector meanSquaredDiffAttrs = sumSquaredDiffAttrs.times(inverseLength);

        DenseVector varianceAttrs = new DenseVector(length);
        for (int i = 0; i < length; i++) {
            varianceAttrs.set(i, Math.sqrt(meanSquaredDiffAttrs.get(i)));
        }

        return varianceAttrs;
    }

    public static int minDistance(List<Integer> ints1, List<Integer> ints2) {
        int len1 = ints1.size();
        int len2 = ints2.size();

        // len1+1, len2+1, because finally return dp[len1][len2]
        int[][] dp = new int[len1 + 1][len2 + 1];

        for (int i = 0; i <= len1; i++) {
            dp[i][0] = i;
        }

        for (int j = 0; j <= len2; j++) {
            dp[0][j] = j;
        }

        //iterate though, and check last int
        for (int i = 0; i < len1; i++) {
            Integer int1 = ints1.get(i);
            for (int j = 0; j < len2; j++) {
                Integer int2 = ints2.get(j);

                //if last two chars equal
                if (int1.equals(int2)) {
                    //update dp value for +1 length
                    dp[i + 1][j + 1] = dp[i][j];
                } else {
                    int replace = dp[i][j] + 1;
                    int insert = dp[i][j + 1] + 1;
                    int delete = dp[i + 1][j] + 1;

                    int min = replace > insert ? insert : replace;
                    min = delete > min ? min : delete;
                    dp[i + 1][j + 1] = min;
                }
            }
        }

        return dp[len1][len2];
    }

    public static String indexesToAttributes(List<Integer> indexes, AttributeLabeledDataSet attributeLabeledDataSet) {
        boolean isFirst = true;
        StringBuilder builder = new StringBuilder("[");
        for (Integer attrIndex : indexes) {
            if (!isFirst) {
                builder.append(", ");
            } else {
                isFirst = false;
            }
            builder.append(attributeLabeledDataSet.getAttributeName(attrIndex));
        }
        return builder.append("]").toString();
    }

    public static List<Double> arrayToList(double[] array) {
        ArrayList<Double> list = new ArrayList<Double>(array.length);
        for (double a : array) {
            list.add(a);
        }
        return list;
    }
}
