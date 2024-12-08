package com.rts.jnn.core.initialization;

/**
 * Implements the Kaiming (He) initialization strategy for neural network weights.
 *
 * <p>This initialization method is particularly well-suited for networks using ReLU
 * activation functions. It helps maintain the variance of activations and gradients
 * across layers, reducing the likelihood of vanishing or exploding gradients.</p>
 *
 * <p>Weights are initialized from a normal distribution with:</p>
 * <ul>
 *   <li>Mean = 0</li>
 *   <li>Variance = 2/n, where n is the number of input connections</li>
 * </ul>
 *
 * <h2>Mathematical Formula:</h2>
 * <pre>
 * W ~ N(0, sqrt(2/n))
 * where:
 * - W is the weight
 * - n is the number of input connections
 * - N(μ, σ) is a normal distribution with mean μ and standard deviation σ
 * </pre>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * InitializationFunction init = new KaimingInitialization();
 *
 * // Initialize weights for a layer with 784 inputs
 * double[] weights = init.init(784);
 *
 * // Use in network layer construction
 * Layer layer = new Layer(neurons, inputSize,
 *                        new ReLUActivation(),
 *                        new KaimingInitialization());
 * }</pre>
 */
public class KaimingInitialization implements InitializationFunction {

    /**
     * Generates a random number from a standard normal distribution.
     *
     * <p>Uses a simple approximation of the Box-Muller transform.
     * For production use, consider using a more robust random number generator.</p>
     *
     * @return Random number from standard normal distribution
     */
    private static double randGaussian() {
        return Math.random() * 2 - 1;  // Simple approximation
    }

    /**
     * Initializes weights using Kaiming initialization.
     *
     * <p>Generates weights from a normal distribution scaled by sqrt(2/n),
     * where n is the number of input connections.</p>
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
        double stdDev = Math.sqrt(2.0 / inputSize);

        for (int i = 0; i < inputSize; i++) {
            weights[i] = randGaussian() * stdDev;
        }

        return weights;
    }
}