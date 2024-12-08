package com.rts.jnn.core.activation;

/**
 * Implements the Hyperbolic Tangent (tanh) activation function.
 *
 * <p>The tanh function is a scaled and shifted version of the sigmoid, outputting values
 * in the range (-1, 1). It's defined as: tanh(x) = (e^x - e^-x) / (e^x + e^-x)
 * This function provides:</p>
 *
 * <ul>
 *   <li>Zero-centered outputs (-1 to 1 range)</li>
 *   <li>Stronger gradients compared to sigmoid</li>
 *   <li>Symmetric around the origin</li>
 *   <li>Smooth, S-shaped curve</li>
 * </ul>
 *
 * <h2>Mathematical Properties:</h2>
 * <ul>
 *   <li>Function: f(x) = tanh(x)</li>
 *   <li>Derivative: f'(x) = 1 - tanh²(x)</li>
 *   <li>Range: (-1, 1)</li>
 *   <li>Centered at: (0, 0)</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * ActivationFunction activation = new TanhActivation();
 *
 * // Function examples
 * double output1 = activation.activate(0.0);   // Returns 0.0
 * double output2 = activation.activate(2.0);   // Returns ~0.964
 * double output3 = activation.activate(-2.0);  // Returns ~-0.964
 *
 * // Derivative examples
 * double deriv1 = activation.derivative(0.0);  // Returns 1.0
 * double deriv2 = activation.derivative(0.5);  // Returns 0.75
 * }</pre>
 *
 * <p><b>Note:</b> While similar to sigmoid, tanh often performs better in practice
 * due to its zero-centered output and stronger gradients.</p>
 */
public class TanhActivation implements ActivationFunction {

    /**
     * Computes the hyperbolic tangent (tanh) function.
     *
     * <p>Maps any real number to the range (-1,1) in a smooth, S-shaped curve
     * that's symmetric around the origin.</p>
     *
     * @param x Input value
     * @return Tanh output in range (-1,1)
     */
    @Override
    public double activate(double x) {
        return Math.tanh(x);
    }

    /**
     * Computes the derivative of tanh: f'(x) = 1 - tanh²(x)
     *
     * <p>The derivative is related to the original tanh value:
     * if y = tanh(x), then dy/dx = 1 - y²</p>
     *
     * @param x The tanh output value
     * @return The derivative value
     */
    @Override
    public double derivative(double x) {
        double tanh = Math.tanh(x);
        return 1 - (tanh * tanh);
    }
}