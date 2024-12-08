package com.rts.jnn.core.activation;

/**
 * Implements the Leaky ReLU (Rectified Linear Unit) activation function.
 *
 * <p>Leaky ReLU is defined as:</p>
 * <ul>
 *   <li>f(x) = x for x > 0</li>
 *   <li>f(x) = αx for x ≤ 0, where α is a small constant (typically 0.01)</li>
 * </ul>
 *
 * <p>It addresses the "dying ReLU" problem by allowing a small, non-zero gradient
 * when the input is negative. Key benefits include:</p>
 *
 * <ul>
 *   <li>Prevents dead neurons (unlike standard ReLU)</li>
 *   <li>Allows for negative values with reduced magnitude</li>
 *   <li>Maintains most benefits of standard ReLU</li>
 *   <li>Simple and computationally efficient</li>
 * </ul>
 *
 * <h2>Mathematical Properties:</h2>
 * <ul>
 *   <li>Function: f(x) = max(αx, x)</li>
 *   <li>Derivative: f'(x) = 1 if x > 0, α if x ≤ 0</li>
 *   <li>Range: (-∞, ∞)</li>
 *   <li>Slope for negative values: α (default 0.01)</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * ActivationFunction activation = new LeakyReLUActivation(0.01);
 *
 * // Function examples
 * double output1 = activation.activate(2.0);   // Returns 2.0
 * double output2 = activation.activate(-2.0);  // Returns -0.02
 *
 * // Derivative examples
 * double deriv1 = activation.derivative(2.0);  // Returns 1.0
 * double deriv2 = activation.derivative(-2.0); // Returns 0.01
 * }</pre>
 */
public class LeakyReLUActivation implements ActivationFunction {

    private final double alpha;

    /**
     * Creates a new Leaky ReLU activation function with specified slope for negative values.
     *
     * @param alpha Slope for negative inputs (typically 0.01)
     * @throws IllegalArgumentException if alpha is negative or greater than 1
     */
    public LeakyReLUActivation(double alpha) {
        if (alpha < 0 || alpha >= 1) {
            throw new IllegalArgumentException("Alpha must be between 0 and 1");
        }
        this.alpha = alpha;
    }

    /**
     * Creates a new Leaky ReLU activation function with default slope of 0.01.
     */
    public LeakyReLUActivation() {
        this(0.01);
    }

    /**
     * Computes Leaky ReLU activation.
     *
     * <p>Returns x for positive values, αx for negative values.</p>
     *
     * @param x Input value
     * @return x if x > 0, αx otherwise
     */
    @Override
    public double activate(double x) {
        return x > 0 ? x : alpha * x;
    }

    /**
     * Computes Leaky ReLU derivative.
     *
     * <p>Returns 1 for positive inputs, α for negative inputs.</p>
     *
     * @param x Input value
     * @return 1 if x > 0, α otherwise
     */
    @Override
    public double derivative(double x) {
        return x > 0 ? 1.0 : alpha;
    }

    /**
     * Gets the alpha value (negative slope) used by this activation function.
     *
     * @return The alpha value
     */
    public double getAlpha() {
        return alpha;
    }
}