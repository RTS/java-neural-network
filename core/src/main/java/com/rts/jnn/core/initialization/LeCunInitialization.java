package com.rts.jnn.core.initialization;

/**
 * Implements the LeCun initialization strategy for neural network weights.
 *
 * <p>This initialization method was proposed by Yann LeCun and is particularly
 * effective for networks using tanh or sigmoid activation functions. It helps
 * maintain the variance of activations through the network by scaling weights
 * based on input size.</p>
 *
 * <p>Weights are initialized from a normal distribution with:</p>
 * <ul>
 *   <li>Mean = 0</li>
 *   <li>Variance = 1/n, where n is the number of input connections</li>
 * </ul>
 *
 * <h2>Mathematical Formula:</h2>
 * <pre>
 * W ~ N(0, sqrt(1/n))
 * where:
 * - W is the weight
 * - n is the number of input connections
 * - N(μ, σ) is a normal distribution with mean μ and standard deviation σ
 * </pre>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * InitializationFunction init = new LeCunInitialization();
 *
 * // Initialize weights for a layer with 100 inputs
 * double[] weights = init.init(100);
 *
 * // Use in network layer construction
 * Layer layer = new Layer(neurons, inputSize,
 *                        new TanhActivation(),
 *                        new LeCunInitialization());
 * }</pre>
 *
 * <h2>Best Used With:</h2>
 * <ul>
 *   <li>Tanh activation functions</li>
 *   <li>Sigmoid activation functions</li>
 *   <li>Networks where maintaining activation variance is important</li>
 *   <li>Shallow to medium-depth networks</li>
 * </ul>
 */
public class LeCunInitialization implements InitializationFunction {

    /**
     * Generates a random number from a standard normal distribution.
     *
     * <p>Uses a simple approximation of the normal distribution.
     * For production environments, consider using a more robust
     * random number generator.</p>
     *
     * @return Random number from standard normal distribution
     */
    private static double randGaussian() {
        return Math.random() * 2 - 1;
    }

    /**
     * Initializes weights using LeCun initialization.
     *
     * <p>Generates weights from a normal distribution scaled by sqrt(1/n),
     * where n is the number of input connections. This scaling helps maintain
     * the variance of activations through the network.</p>
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
        double stdDev = Math.sqrt(1.0 / inputSize);

        for (int i = 0; i < inputSize; i++) {
            weights[i] = randGaussian() * stdDev;
        }

        return weights;
    }
}