package com.rts.jnn.core.activation;

/**
 * Implements the Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
 *
 * <p>The Sigmoid function is a smooth, S-shaped curve that maps any input to a value
 * between 0 and 1. It's particularly useful for:</p>
 *
 * <ul>
 *   <li>Binary classification problems</li>
 *   <li>Output layers where probabilities are needed</li>
 *   <li>Networks dealing with binary or normalized data</li>
 *   <li>Cases where smooth, bounded outputs are desired</li>
 * </ul>
 *
 * <h2>Mathematical Properties:</h2>
 * <ul>
 *   <li>Function: f(x) = 1 / (1 + e^(-x))</li>
 *   <li>Derivative: f'(x) = f(x) * (1 - f(x))</li>
 *   <li>Range: (0, 1)</li>
 *   <li>Centered at: (0, 0.5)</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * ActivationFunction activation = new SigmoidActivation();
 *
 * // Function examples
 * double output1 = activation.activate(0.0);   // Returns 0.5
 * double output2 = activation.activate(2.0);   // Returns 0.8808
 * double output3 = activation.activate(-2.0);  // Returns 0.1192
 *
 * // Derivative examples
 * double deriv1 = activation.derivative(0.5);  // Returns 0.25
 * double deriv2 = activation.derivative(0.8);  // Returns 0.16
 * }</pre>
 *
 * <p><b>Note:</b> The derivative input should be the output of the sigmoid function,
 * not the original input value. This optimization is possible because the derivative
 * can be expressed in terms of the function output.</p>
 */
public class SigmoidActivation implements ActivationFunction {

    /**
     * Computes the sigmoid function: 1 / (1 + e^(-x))
     *
     * <p>Maps any real number to the range (0,1) in a smooth, S-shaped curve.</p>
     *
     * @param x Input value
     * @return Sigmoid output in range (0,1)
     */
    @Override
    public double activate(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    /**
     * Computes the derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
     *
     * <p><b>Important:</b> The input x should be f(x), the output of activate().
     * This allows for more efficient computation as we don't need to recompute
     * the sigmoid function.</p>
     *
     * @param x The sigmoid output value (not the original input)
     * @return The derivative value
     */
    @Override
    public double derivative(double x) {
        return x * (1 - x);
    }
}