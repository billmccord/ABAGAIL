package shared;

/**
 * A convergence trainer trains a network
 * until convergence, using another trainer
 *
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class ConvergenceTrainer implements IterationTrainer {
    /**
     * The default threshold
     */
    private static final double THRESHOLD = 1E-10;
    /**
     * The maxium number of iterations
     */
    private static final int MAX_ITERATIONS = 500;

    /**
     * The trainer
     */
    private Trainer trainer;

    /**
     * The threshold
     */
    private double threshold;

    /**
     * The number of iterations trained
     */
    private int iterations;

    /**
     * The maximum number of iterations to use
     */
    private int maxIterations;

    private int convergedIterations = 0;

    private int confirmationIterations = 0;

    /**
     * Create a new convergence trainer
     *
     * @param trainer       the trainer to use
     * @param threshold     the error threshold
     * @param maxIterations the maximum iterations
     */
    public ConvergenceTrainer(Trainer trainer,
                              double threshold, int maxIterations) {
        this(trainer, threshold, maxIterations, 0);
    }

    /**
     * Create a new convergence trainer
     *
     * @param trainer       the trainer to use
     * @param threshold     the error threshold
     * @param maxIterations the maximum iterations
     * @param confirmationIterations the number of iterations to use to confirm that we have indeed converged.
     */
    public ConvergenceTrainer(Trainer trainer,
                              double threshold, int maxIterations,
                              int confirmationIterations) {
        this.trainer = trainer;
        this.threshold = threshold;
        this.maxIterations = maxIterations;
        this.confirmationIterations = confirmationIterations;
    }

    /**
     * Create a new convergence trainer
     *
     * @param trainer the trainer to use
     */
    public ConvergenceTrainer(Trainer trainer) {
        this(trainer, THRESHOLD, MAX_ITERATIONS);
    }

    /**
     * @see Trainer#train()
     */
    public double train() {
        double errorDiff;
        double lastError;
        double error = Double.MAX_VALUE;
        convergedIterations = 0;
        do {
            iterations++;
            lastError = error;
            error = trainer.train();
            errorDiff = Math.abs(error - lastError);
            if (errorDiff <= threshold) {
                convergedIterations++;
            } else {
                convergedIterations = 0;
            }
        } while ((Math.abs(error - lastError) > threshold || convergedIterations < confirmationIterations)
                && iterations < maxIterations);
        return error;
    }

    /**
     * Get the number of iterations used
     *
     * @return the number of iterations
     */
    @Override
    public int getIterations() {
        return iterations - convergedIterations;
    }
}
