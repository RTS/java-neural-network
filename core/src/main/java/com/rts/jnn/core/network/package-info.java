/**
 * Neural network implementation package providing Morse code translation capabilities.
 *
 * <p>This package contains the core components for building, training, and using neural networks
 * specifically designed for Morse code translation. It provides a flexible architecture that
 * supports:</p>
 *
 * <ul>
 *   <li>Configurable network topology with multiple layers</li>
 *   <li>Various activation functions</li>
 *   <li>Multiple weight initialization strategies</li>
 *   <li>Adjustable learning rates with decay options</li>
 * </ul>
 *
 * <h2>Key Components:</h2>
 * <ul>
 *   <li>{@link com.rts.jnn.core.network.NeuralNetwork} - Main network implementation</li>
 *   <li>{@link com.rts.jnn.core.network.Layer} - Network layer abstraction</li>
 *   <li>{@link com.rts.jnn.core.network.Neuron} - Individual neuron implementation</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Create network with learning rate 0.5
 * NeuralNetwork network = new NeuralNetwork(0.5);
 *
 * // Add layers
 * network.addLayer(5, new SigmoidActivation(), new XavierInitialization());
 * network.addLayer(20, new SigmoidActivation(), new XavierInitialization());
 * network.addLayer(36, new SigmoidActivation(), new XavierInitialization());
 *
 * // Train network
 * network.train(inputs, targets);
 * }</pre>
 *
 * @see com.rts.jnn.core.activation
 * @see com.rts.jnn.core.initialization
 * @see com.rts.jnn.core.decay
 */
package com.rts.jnn.core.network;