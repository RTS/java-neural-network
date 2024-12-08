package com.rts.jnn.core.activation;

/**
 * Implements the Linear (Identity) activation function.
 *
 * <p>This activation function returns the input value unchanged: f(x) = x.
 * Its derivative is always 1: f'(x) = 1. Linear activation is useful for:</p>
 *
 * <ul>
 *   <li>Regression problems where unbounded outputs are desired</li>
 *   <li>Testing or debugging neural networks</li>
 *   <li>Final layer in some regression networks</li>
 * </ul>
 *
 * <h2>Mathematical Definition:</h2>
 * <ul>
 *   <li>Function: f(x) = x</li>
 *   <li>Derivative: f'(x) = 1</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * ActivationFunction activation = new LinearActivation();
 *
 * // Function examples
 * double output1 = activation.activate(0.5);  // Returns 0.5
 * double output2 = activation.activate(-1.0); // Returns -1.0
 *
 * // Derivative examples
 * double deriv1 = activation.derivative(0.5);  // Returns 1.0
 * double deriv2 = activation.derivative(-1.0); // Returns 1.0
 * }</pre>
 *
 * <p><b>Note:</b> While simple, linear activation functions don't allow the network
 * to learn non-linear patterns. They're typically only used in specific scenarios
 * where linear behavior is desired.</p>
 *
 * @see ActivationFunction
 * @see com.rts.jnn.core.network.Neuron
 */
public class LinearActivation implements ActivationFunction {

    /**
     * Computes the linear activation function output.
     *
     * <p>Returns the input value unchanged (identity function).</p>
     *
     * @param x Input value
     * @return Same value as input (f(x) = x)
     */
    @Override
    public double activate(double x) {
        return x;
    }

    /**
     * Computes the derivative of the linear activation function.
     *
     * <p>Returns 1.0 for all inputs as the derivative of f(x) = x is 1.</p>
     *
     * @param x Input value (not used in computation)
     * @return 1.0 (constant derivative)
     */
    @Override
    public double derivative(double x) {
        return 1.0;
    }
}