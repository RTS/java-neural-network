package com.rts.jnn.core.activation;

/**
 * Defines the contract for activation functions used in neural network neurons.
 *
 * <p>Activation functions introduce non-linearity into neural networks by transforming
 * the weighted sum of inputs into the neuron's output. They are crucial for the
 * network's ability to learn complex patterns and relationships.</p>
 *
 * <h2>Available Implementations:</h2>
 * <ul>
 *   <li>{@link BentIdentityActivation} - Smooth alternative with non-zero gradients</li>
 *   <li>{@link ELUActivation} - Exponential Linear Unit with negative values</li>
 *   <li>{@link LeakyReLUActivation} - ReLU variant with small negative slope</li>
 *   <li>{@link LinearActivation} - Simple identity function</li>
 *   <li>{@link ReLUActivation} - Rectified Linear Unit</li>
 *   <li>{@link SigmoidActivation} - S-shaped function (0 to 1)</li>
 *   <li>{@link SwishActivation} - Self-gated activation</li>
 *   <li>{@link TanhActivation} - Hyperbolic tangent (-1 to 1)</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // Create an activation function
 * ActivationFunction sigmoid = new SigmoidActivation();
 *
 * // Forward propagation
 * double output = sigmoid.activate(weightedSum);
 *
 * // Backpropagation
 * double derivative = sigmoid.derivative(output);
 * }</pre>
 *
 * <h2>Implementation Requirements:</h2>
 * <ul>
 *   <li>Must provide both activation and its derivative</li>
 *   <li>Should be thread-safe (stateless)</li>
 *   <li>Should handle numerical stability</li>
 *   <li>Must be continuous and differentiable</li>
 * </ul>
 *
 * <h2>Selection Guidelines:</h2>
 * <p>Choose activation functions based on:</p>
 * <ul>
 *   <li>Network depth and architecture</li>
 *   <li>Type of problem (classification, regression)</li>
 *   <li>Desired output range</li>
 *   <li>Training stability requirements</li>
 * </ul>
 *
 * @see com.rts.jnn.core.network.Neuron
 * @see com.rts.jnn.core.network.Layer
 */
public interface ActivationFunction {

    /**
     * Computes the activation function output.
     *
     * <p>Takes a weighted sum input and transforms it through the activation function.
     * The specific transformation depends on the implementation.</p>
     *
     * @param x Input value (typically weighted sum of neuron inputs plus bias)
     * @return Transformed output value
     */
    double activate(double x);

    /**
     * Computes the derivative of the activation function.
     *
     * <p>The derivative is used during backpropagation to calculate gradients.
     * Note that some implementations may expect the input to be the output
     * of the activation function rather than the original input.</p>
     *
     * @param x Input value (implementation specific)
     * @return Derivative value at x
     */
    double derivative(double x);
}