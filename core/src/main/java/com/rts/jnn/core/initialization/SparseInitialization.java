package com.rts.jnn.core.initialization;

/**
 * Implements sparse weight initialization for neural networks.
 *
 * <p>This initialization strategy creates sparse weight matrices by initializing
 * only a portion of weights with non-zero values. The sparsity is controlled by
 * a sparsity level parameter (default 0.5), which determines the probability
 * of a weight being non-zero.</p>
 *
 * <p>Weight initialization follows:</p>
 * <ul>
 *   <li>Probability(non-zero) = sparsityLevel</li>
 *   <li>Non-zero weights ~ U(-1, 1)</li>
 *   <li>Zero weights = 0.0</li>
 * </ul>
 *
 * <h2>Mathematical Properties:</h2>
 * <pre>
 * For each weight w:
 * w = 0.0                  with probability (1 - sparsityLevel)
 * w ~ Uniform(-1, 1)       with probability sparsityLevel
 * </pre>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * InitializationFunction init = new SparseInitialization();
 *
 * // Initialize weights for a layer with 100 inputs
 * double[] weights = init.init(100);
 *
 * // Use in network layer construction
 * Layer layer = new Layer(neurons, inputSize,
 *                        activation,
 *                        new SparseInitialization());
 * }</pre>
 *
 * <h2>Advantages:</h2>
 * <ul>
 *   <li>Reduces initial network connectivity</li>
 *   <li>Can improve training efficiency</li>
 *   <li>May help prevent overfitting</li>
 *   <li>Useful for large network architectures</li>
 * </ul>
 */
public class SparseInitialization implements InitializationFunction {

    /**
     * Default sparsity level
     */
    private static final double DEFAULT_SPARSITY_LEVEL = 0.5;

    private final double sparsityLevel;

    /**
     * Creates a new sparse initialization with default sparsity level (0.5).
     */
    public SparseInitialization() {
        this(DEFAULT_SPARSITY_LEVEL);
    }

    /**
     * Creates a new sparse initialization with specified sparsity level.
     *
     * @param sparsityLevel Probability of non-zero weights (0,1)
     * @throws IllegalArgumentException if sparsityLevel not in (0,1)
     */
    public SparseInitialization(double sparsityLevel) {
        if (sparsityLevel <= 0 || sparsityLevel >= 1) {
            throw new IllegalArgumentException(
                    "Sparsity level must be between 0 and 1");
        }
        this.sparsityLevel = sparsityLevel;
    }

    /**
     * Initializes weights using sparse initialization.
     *
     * <p>Creates a weight array where some weights are initialized to zero
     * based on the sparsity level, and others are randomly initialized
     * between -1 and 1.</p>
     *
     * @param inputSize Number of input connections to the neuron
     * @return Array of initialized weights
     * @throws IllegalArgumentException if inputSize is less than 1
     */
    @Override
    public double[] init(int inputSize) {
        if (inputSize < 1) {
            throw new IllegalArgumentException("Input size must be at least 1");
        }

        double[] weights = new double[inputSize];

        for (int i = 0; i < inputSize; i++) {
            if (Math.random() < sparsityLevel) {
                // Initialize non-zero weight uniformly between -1 and 1
                weights[i] = Math.random() * 2 - 1;
            } else {
                // Initialize to zero
                weights[i] = 0.0;
            }
        }

        return weights;
    }

    /**
     * Gets the sparsity level used in this initialization.
     *
     * @return The sparsity level
     */
    public double getSparsityLevel() {
        return sparsityLevel;
    }
}