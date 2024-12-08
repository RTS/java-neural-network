package com.rts.jnn.core.activation;

/**
 * Implements the Swish activation function: f(x) = x * sigmoid(x)
 *
 * <p>Swish is a self-gated activation function proposed by Google Brain researchers.
 * It's defined as x * sigmoid(x) and has been shown to outperform ReLU in some deep
 * architectures. Key characteristics include:</p>
 *
 * <ul>
 *   <li>Smooth, non-monotonic function</li>
 *   <li>Bounded below but unbounded above</li>
 *   <li>Has a slight dip below zero for negative values</li>
 *   <li>Approaches ReLU for large positive values</li>
 * </ul>
 *
 * <h2>Mathematical Properties:</h2>
 * <ul>
 *   <li>Function: f(x) = x * σ(x), where σ(x) is sigmoid</li>
 *   <li>Derivative: f'(x) = f(x) + σ(x)(1 - f(x))</li>
 *   <li>Range: Approximately (-0.278, ∞)</li>
 *   <li>Non-monotonic near x = 0</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * ActivationFunction activation = new SwishActivation();
 *
 * // Function examples
 * double output1 = activation.activate(2.0);   // Returns ~1.962
 * double output2 = activation.activate(0.0);   // Returns 0.0
 * double output3 = activation.activate(-2.0);  // Returns ~-0.238
 *
 * // Derivative examples
 * double deriv1 = activation.derivative(1.0);  // Returns ~1.389
 * double deriv2 = activation.derivative(-1.0); // Returns ~0.352
 * }</pre>
 *
 * <p><b>Performance Note:</b> Swish is more computationally expensive than ReLU
 * due to the sigmoid calculation, but may provide better accuracy in deep networks.</p>
 */
public class SwishActivation implements ActivationFunction {

    /**
     * Computes the Swish activation: x * sigmoid(x)
     *
     * <p>Combines the input with its sigmoid activation to create a smooth,
     * non-monotonic function that allows some negative values to pass through.</p>
     *
     * @param x Input value
     * @return Swish activation output
     */
    @Override
    public double activate(double x) {
        return x * (1.0 / (1.0 + Math.exp(-x)));
    }

    /**
     * Computes the derivative of Swish.
     *
     * <p>The derivative is: f'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
     * This implementation computes it efficiently by reusing the sigmoid value.</p>
     *
     * @param x Input value
     * @return Derivative of Swish at x
     */
    @Override
    public double derivative(double x) {
        double sigmoid = 1.0 / (1.0 + Math.exp(-x));
        return sigmoid + x * sigmoid * (1 - sigmoid);
    }
}