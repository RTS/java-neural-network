package com.rts.jnn.core.activation;

/**
 * Implements the ReLU (Rectified Linear Unit) activation function.
 *
 * <p>ReLU is defined as f(x) = max(0,x). It's one of the most widely used activation
 * functions in modern neural networks because it:</p>
 *
 * <ul>
 *   <li>Reduces the vanishing gradient problem</li>
 *   <li>Provides faster training compared to sigmoid/tanh</li>
 *   <li>Introduces true sparsity in activations</li>
 *   <li>Is computationally efficient</li>
 * </ul>
 *
 * <h2>Mathematical Definition:</h2>
 * <ul>
 *   <li>Function: f(x) = max(0,x)</li>
 *   <li>Derivative: f'(x) = 1 if x > 0, 0 if x ≤ 0</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * ActivationFunction activation = new ReLUActivation();
 *
 * // Function examples
 * double output1 = activation.activate(2.0);   // Returns 2.0
 * double output2 = activation.activate(-1.0);  // Returns 0.0
 *
 * // Derivative examples
 * double deriv1 = activation.derivative(2.0);  // Returns 1.0
 * double deriv2 = activation.derivative(-1.0); // Returns 0.0
 * }</pre>
 *
 * <p><b>Note:</b> While ReLU can suffer from "dying ReLU" problem where neurons
 * can get stuck in a permanently inactive state, it's still highly effective
 * in practice and is the default choice for many deep learning applications.</p>
 */
public class ReLUActivation implements ActivationFunction {

    /**
     * Computes ReLU activation: max(0,x)
     *
     * <p>Returns the input value if positive, otherwise returns 0.</p>
     *
     * @param x Input value
     * @return x if x > 0, otherwise 0
     */
    @Override
    public double activate(double x) {
        return Math.max(0, x);
    }

    /**
     * Computes ReLU derivative.
     *
     * <p>Returns 1 for positive inputs, 0 otherwise.</p>
     *
     * @param x Input value
     * @return 1 if x > 0, 0 otherwise
     */
    @Override
    public double derivative(double x) {
        return x > 0 ? 1 : 0;
    }
}