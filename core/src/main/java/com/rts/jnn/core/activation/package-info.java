/**
 * Provides activation functions for neural network neurons and layers.
 *
 * <p>This package contains implementations of various activation functions commonly used
 * in neural networks. Each implementation provides both the activation calculation and
 * its derivative for use in backpropagation training.</p>
 *
 * <h2>Available Activation Functions:</h2>
 * <ul>
 *   <li>{@link com.rts.jnn.core.activation.BentIdentityActivation} - Smooth alternative with non-zero gradients everywhere</li>
 *   <li>{@link com.rts.jnn.core.activation.ELUActivation} - Exponential Linear Unit with smooth negative values</li>
 *   <li>{@link com.rts.jnn.core.activation.LeakyReLUActivation} - ReLU variant allowing small negative gradients</li>
 *   <li>{@link com.rts.jnn.core.activation.LinearActivation} - Simple identity function</li>
 *   <li>{@link com.rts.jnn.core.activation.ReLUActivation} - Rectified Linear Unit, standard in deep learning</li>
 *   <li>{@link com.rts.jnn.core.activation.SigmoidActivation} - Classic S-shaped function (0 to 1 range)</li>
 *   <li>{@link com.rts.jnn.core.activation.SwishActivation} - Self-gated activation function</li>
 *   <li>{@link com.rts.jnn.core.activation.TanhActivation} - Hyperbolic tangent (-1 to 1 range)</li>
 * </ul>
 *
 * <h2>Activation Function Families:</h2>
 * <p>The functions can be grouped into several families:</p>
 *
 * <h3>ReLU Family:</h3>
 * <ul>
 *   <li>ReLU - Standard rectified linear unit</li>
 *   <li>Leaky ReLU - Prevents "dying ReLU" problem</li>
 *   <li>ELU - Smooth negative values with exponential behavior</li>
 * </ul>
 *
 * <h3>Sigmoid Family:</h3>
 * <ul>
 *   <li>Sigmoid - Bounded between 0 and 1</li>
 *   <li>Tanh - Bounded between -1 and 1</li>
 * </ul>
 *
 * <h3>Modern Variants:</h3>
 * <ul>
 *   <li>Swish - Self-gated activation</li>
 *   <li>Bent Identity - Smooth alternative with good gradient properties</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Create an activation function
 * ActivationFunction relu = new ReLUActivation();
 * ActivationFunction sigmoid = new SigmoidActivation();
 * ActivationFunction elu = new ELUActivation(1.0);
 *
 * // Use in forward propagation
 * double output = relu.activate(2.0);
 *
 * // Use derivative for backpropagation
 * double derivative = relu.derivative(output);
 * }</pre>
 *
 * <h2>Selection Guidelines:</h2>
 * <ul>
 *   <li>Deep Networks: ReLU, Leaky ReLU, ELU</li>
 *   <li>Binary Classification: Sigmoid</li>
 *   <li>Feature Scaling: Tanh</li>
 *   <li>Modern Deep Learning: Swish</li>
 *   <li>Regression Tasks: Linear, Bent Identity</li>
 * </ul>
 *
 * <h2>Common Properties:</h2>
 * <ul>
 *   <li>All implementations are thread-safe</li>
 *   <li>All functions are continuous</li>
 *   <li>All provide derivatives for backpropagation</li>
 *   <li>No mutable state</li>
 * </ul>
 *
 * @see com.rts.jnn.core.activation.ActivationFunction
 * @see com.rts.jnn.core.network.Neuron
 * @see com.rts.jnn.core.network.Layer
 */
package com.rts.jnn.core.activation;