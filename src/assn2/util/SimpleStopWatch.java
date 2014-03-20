package assn2.util;

/**
 * Created by bill on 3/14/14.
 */
public class SimpleStopWatch {
    private double start = System.nanoTime();

    public void start() {
        start = System.nanoTime();
    }

    public double stop() {
        return (System.nanoTime() - start) / Math.pow(10,9);
    }
}
