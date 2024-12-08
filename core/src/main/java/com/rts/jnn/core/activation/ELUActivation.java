package com.rts.jnn.core.activation;

/**
 * Implements the Exponential Linear Unit (ELU) activation function.
 *
 * <p>ELU is defined as:</p>
 * <ul>
 *   <li>f(x) = x for x > 0</li>
 *   <li>f(x) = α(e^x - 1) for x ≤ 0, where α is a positive parameter (typically 1.0)</li>
 * </ul>
 *
 * <p>ELU combines the benefits of ReLU with several advantages:</p>
 * <ul>
 *   <li>Smooth transition around zero (differentiable everywhere)</li>
 *   <li>Negative values allowed (helps push mean activations toward zero)</li>
 *   <li>Natural gradient restoration through exponential decay</li>
 *   <li>More robust to noise than ReLU and variants</li>
 * </ul>
 *
 * <h2>Mathematical Properties:</h2>
 * <ul>
 *   <li>Function: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0</li>
 *   <li>Derivative: f'(x) = 1 if x > 0, f(x) + α if x ≤ 0</li>
 *   <li>Range: (-α, ∞)</li>
 *   <li>Slope at x = 0: 1 (continuous derivative)</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * ActivationFunction activation = new ELUActivation(1.0);
 *
 * // Function examples
 * double output1 = activation.activate(2.0);   // Returns 2.0
 * double output2 = activation.activate(-2.0);  // Returns -0.865 (≈ -0.865)
 *
 * // Derivative examples
 * double deriv1 = activation.derivative(2.0);   // Returns 1.0
 * double deriv2 = activation.derivative(-2.0);  // Returns 0.135
 * }</pre>
 */
public class ELUActivation implements ActivationFunction {

    private final double alpha;

    /**
     * Creates a new ELU activation function with specified alpha parameter.
     *
     * @param alpha Scale for negative values (typically 1.0)
     * @throws IllegalArgumentException if alpha is not positive
     */
    public ELUActivation(double alpha) {
        if (alpha <= 0) {
            throw new IllegalArgumentException("Alpha must be positive");
        }
        this.alpha = alpha;
    }

    /**
     * Creates a new ELU activation function with default alpha of 1.0.
     */
    public ELUActivation() {
        this(1.0);
    }

    /**
     * Computes ELU activation.
     *
     * <p>For positive inputs, behaves like identity function.
     * For negative inputs, produces smooth negative values via exponential decay.</p>
     *
     * @param x Input value
     * @return x if x > 0, α(e^x - 1) otherwise
     */
    @Override
    public double activate(double x) {
        return x > 0 ? x : alpha * (Math.exp(x) - 1);
    }

    /**
     * Computes ELU derivative.
     *
     * <p>The derivative is continuous at x = 0, unlike ReLU and its variants.</p>
     *
     * @param x Input value
     * @return 1 if x > 0, α * e^x otherwise
     */
    @Override
    public double derivative(double x) {
        return x > 0 ? 1.0 : alpha * Math.exp(x);
    }

    /**
     * Gets the alpha parameter used by this activation function.
     *
     * @return The alpha value
     */
    public double getAlpha() {
        return alpha;
    }
}