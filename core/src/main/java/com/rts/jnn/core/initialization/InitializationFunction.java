package com.rts.jnn.core.initialization;

/**
 * Defines the contract for weight initialization strategies in neural networks.
 *
 * <p>Weight initialization is crucial for neural network training as it affects:</p>
 * <ul>
 *   <li>Training convergence speed</li>
 *   <li>Final model performance</li>
 *   <li>Gradient flow through the network</li>
 *   <li>Probability of vanishing/exploding gradients</li>
 * </ul>
 *
 * <h2>Available Implementations:</h2>
 * <ul>
 *   <li>{@link KaimingInitialization} - For ReLU networks: N(0, sqrt(2/n))</li>
 *   <li>{@link LeCunInitialization} - For tanh networks: N(0, sqrt(1/n))</li>
 *   <li>{@link ScalingInitialization} - Conservative approach: N(0, sqrt(0.2/n))</li>
 *   <li>{@link SparseInitialization} - Sparse weights with controlled density</li>
 *   <li>{@link XavierInitialization} - Classic method: U(-sqrt(6/n), sqrt(6/n))</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // Create an initialization strategy
 * InitializationFunction init = new XavierInitialization();
 *
 * // Initialize weights for a layer
 * double[] weights = init.init(inputSize);
 *
 * // Use in layer construction
 * Layer layer = new Layer(
 *     neurons,
 *     inputSize,
 *     new TanhActivation(),
 *     init
 * );
 * }</pre>
 *
 * <h2>Implementation Guidelines:</h2>
 * <p>When implementing this interface:</p>
 * <ul>
 *   <li>Ensure thread-safety (implementations should be stateless)</li>
 *   <li>Validate input parameters</li>
 *   <li>Consider numerical stability</li>
 *   <li>Document the mathematical basis</li>
 *   <li>Handle edge cases appropriately</li>
 * </ul>
 *
 * @see com.rts.jnn.core.network.Layer
 * @see com.rts.jnn.core.activation.ActivationFunction
 */
public interface InitializationFunction {

    /**
     * Initializes weights for a neuron or layer.
     *
     * <p>This method should generate an array of weight values based on
     * the initialization strategy's mathematical formula. The specific
     * distribution and scaling of weights depends on the implementation.</p>
     *
     * @param inputSize Number of input connections (fan-in)
     * @return Array of initialized weights
     * @throws IllegalArgumentException if inputSize is less than 1
     */
    double[] init(int inputSize);
}