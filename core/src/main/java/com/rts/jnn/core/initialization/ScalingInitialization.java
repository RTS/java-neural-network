package com.rts.jnn.core.initialization;

/**
 * Implements a scaled variant of weight initialization for neural networks.
 *
 * <p>This initialization method uses a scaling factor of 0.2 to generate weights
 * from a normal distribution. The scale factor helps control the initial magnitude
 * of weights, which can be particularly useful for networks where other initialization
 * methods may produce weights that are too large or too small.</p>
 *
 * <p>Weights are initialized from a normal distribution with:</p>
 * <ul>
 *   <li>Mean = 0</li>
 *   <li>Variance = 0.2/n, where n is the number of input connections</li>
 * </ul>
 *
 * <h2>Mathematical Formula:</h2>
 * <pre>
 * W ~ N(0, sqrt(0.2/n))
 * where:
 * - W is the weight
 * - n is the number of input connections
 * - N(μ, σ) is a normal distribution with mean μ and standard deviation σ
 * </pre>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * InitializationFunction init = new ScalingInitialization();
 *
 * // Initialize weights for a layer with 100 inputs
 * double[] weights = init.init(100);
 *
 * // Use in network layer construction
 * Layer layer = new Layer(neurons, inputSize,
 *                        someActivation,
 *                        new ScalingInitialization());
 * }</pre>
 *
 * <h2>Characteristics:</h2>
 * <ul>
 *   <li>Produces smaller initial weights compared to standard initializations</li>
 *   <li>May help prevent initial saturation in deep networks</li>
 *   <li>Useful when more conservative weight initialization is desired</li>
 *   <li>Can be particularly effective with certain activation functions</li>
 * </ul>
 */
public class ScalingInitialization implements InitializationFunction {

    /**
     * The scaling factor used in the initialization
     */
    private static final double SCALE = 0.2;

    /**
     * Generates a random number from a standard normal distribution.
     *
     * <p>Uses a simple approximation of the normal distribution.
     * For production use, consider using a more robust random number generator.</p>
     *
     * @return Random number from standard normal distribution
     */
    private static double randGaussian() {
        return Math.random() * 2 - 1;
    }

    /**
     * Gets the scaling factor used in the initialization.
     *
     * @return The scaling factor (0.2)
     */
    public static double getScale() {
        return SCALE;
    }

    /**
     * Initializes weights using scaled initialization.
     *
     * <p>Generates weights from a normal distribution scaled by sqrt(0.2/n),
     * where n is the number of input connections. This produces weights that
     * are generally smaller in magnitude than other initialization methods.</p>
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
        double stdDev = Math.sqrt(SCALE / inputSize);

        for (int i = 0; i < inputSize; i++) {
            weights[i] = randGaussian() * stdDev;
        }

        return weights;
    }
}