package assn2.util;

import shared.DataSet;
import shared.DataSetDescription;
import shared.Instance;
import shared.filt.LabelSplitFilter;
import shared.reader.ArffDataSetReader;

public class DataSetUtil {
    private static final String DIR_PREFIX = "src/assn2/data/";

    private static final String DATA_FILE_TEMPLATE = DIR_PREFIX + "%s/nursery-%s.arff";

    private static final String NURSERY_TRAINING_FILE = String.format(DATA_FILE_TEMPLATE, "training", "training");

    private static final String NURSERY_TEST_FILE = String.format(DATA_FILE_TEMPLATE, "test", "test");

    public static DataSet readNurseryTrainingDataSet() {
        return DataSetUtil.readDataSet(NURSERY_TRAINING_FILE);
    }

    public static DataSet readNurseryTestDataSet() {
        return DataSetUtil.readDataSet(NURSERY_TEST_FILE);
    }

    public static DataSet readDataSet(String fileName) {
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
