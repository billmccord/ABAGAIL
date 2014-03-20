package assn2.util;

import shared.writer.CSVWriter;

import java.io.IOException;

/**
 * Created by bill on 3/15/14.
 */
public interface CsvWriteable {
    String[] getHeaders();
    void write(CSVWriter writer) throws IOException;
}
