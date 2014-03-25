package assn2.util;

import shared.DataSet;

import java.util.List;

public class AttributeLabeledDataSet {
    private List<String> attributeNames;
    private DataSet dataSet;

    public AttributeLabeledDataSet(List<String> attributeNames, DataSet dataSet) {
        this.attributeNames = attributeNames;
        this.dataSet = dataSet;
    }

    public DataSet getDataSet() {
        return dataSet;
    }

    public String getAttributeName(int i) {
        return attributeNames.get(i);
    }
}
