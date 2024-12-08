package com.rts.jnn.core.activation;

/**
 * Implements the Bent Identity activation function.
 *
 * <p>The Bent Identity function is defined as:</p>
 * <pre>f(x) = (√(x² + 1) - 1)/2 + x</pre>
 *
 * <p>This function provides several unique characteristics:</p>
 * <ul>
 *   <li>Smooth and continuous everywhere</li>
 *   <li>Non-zero gradients for all inputs</li>
 *   <li>Approximately linear for large positive values</li>
 *   <li>Bent behavior near the origin</li>
 * </ul>
 *
 * <h2>Mathematical Properties:</h2>
 * <ul>
 *   <li>Function: f(x) = (√(x² + 1) - 1)/2 + x</li>
 *   <li>Derivative: f'(x) = x/(2√(x² + 1)) + 1</li>
 *   <li>Range: (-∞, ∞)</li>
 *   <li>Always differentiable</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * ActivationFunction activation = new BentIdentityActivation();
 *
 * // Function examples
 * double output1 = activation.activate(2.0);   // Returns ~2.62
 * double output2 = activation.activate(0.0);   // Returns 0.0
 * double output3 = activation.activate(-2.0);  // Returns ~-1.38
 *
 * // Derivative examples
 * double deriv1 = activation.derivative(2.0);  // Returns ~1.45
 * double deriv2 = activation.derivative(0.0);  // Returns 1.0
 * }</pre>
 */
public class BentIdentityActivation implements ActivationFunction {

    /**
     * Computes the Bent Identity activation function.
     *
     * <p>Returns (√(x² + 1) - 1)/2 + x</p>
     *
     * @param x Input value
     * @return Bent Identity output
     */
    @Override
    public double activate(double x) {
        return (Math.sqrt(x * x + 1) - 1) / 2 + x;
    }

    /**
     * Computes the derivative of the Bent Identity function.
     *
     * <p>Returns x/(2√(x² + 1)) + 1</p>
     *
     * @param x Input value
     * @return Derivative value
     */
    @Override
    public double derivative(double x) {
        return x / (2 * Math.sqrt(x * x + 1)) + 1;
    }
}