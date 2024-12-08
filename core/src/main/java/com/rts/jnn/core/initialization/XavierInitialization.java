package com.rts.jnn.core.initialization;

/**
 * Implements the Xavier (Glorot) initialization strategy for neural network weights.
 *
 * <p>This initialization method is designed to maintain the variance of activations
 * and gradients across layers, particularly effective for networks using tanh or
 * sigmoid activations. Weights are uniformly distributed with variance based on
 * input size.</p>
 *
 * <p>Weights are initialized using a uniform distribution with:</p>
 * <ul>
 *   <li>Range: [-limit, limit]</li>
 *   <li>limit = sqrt(6 / (fanIn + fanOut))</li>
 *   <li>Where fanIn is the number of input connections</li>
 *   <li>For this implementation, fanOut is approximated as 1</li>
 * </ul>
 *
 * <h2>Mathematical Formula:</h2>
 * <pre>
 * W ~ U(-sqrt(6/n), sqrt(6/n))
 * where:
 * - W is the weight
 * - n is the number of input connections + 1
 * - U(a,b) is a uniform distribution between a and b
 * </pre>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * InitializationFunction init = new XavierInitialization();
 *
 * // Initialize weights for a layer with 100 inputs
 * double[] weights = init.init(100);
 *
 * // Use in network layer construction
 * Layer layer = new Layer(neurons, inputSize,
 *                        new TanhActivation(),
 *                        new XavierInitialization());
 * }</pre>
 *
 * <h2>Best Used With:</h2>
 * <ul>
 *   <li>Tanh activation functions</li>
 *   <li>Sigmoid activation functions</li>
 *   <li>Networks of moderate depth</li>
 *   <li>When maintaining activation variance is crucial</li>
 * </ul>
 */
public class XavierInitialization implements InitializationFunction {

    /**
     * Initializes weights using Xavier initialization.
     *
     * <p>Generates weights from a uniform distribution with limits calculated
     * using the Xavier formula. This helps maintain the variance of activations
     * through the network.</p>
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
        // Calculate limit for uniform distribution
        double limit = Math.sqrt(6.0 / (inputSize + 1));

        for (int i = 0; i < inputSize; i++) {
            // Generate uniform random number between -limit and limit
            weights[i] = (Math.random() * 2 * limit) - limit;
        }

        return weights;
    }
}