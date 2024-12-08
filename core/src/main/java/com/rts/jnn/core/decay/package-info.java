/**
 * Provides learning rate decay strategies for neural network training optimization.
 *
 * <p>This package contains various implementations of learning rate decay functions
 * that help optimize the training process by gradually reducing the learning rate
 * over time. Proper learning rate scheduling can significantly improve training
 * stability and final model performance.</p>
 *
 * <h2>Available Decay Functions:</h2>
 * <ul>
 *   <li>{@link com.rts.jnn.core.decay.ExponentialDecay} - Smooth exponential decrease</li>
 *   <li>{@link com.rts.jnn.core.decay.InverseTimeDecay} - Gradual inverse time reduction</li>
 *   <li>{@link com.rts.jnn.core.decay.PolynomialDecay} - Configurable polynomial decay to end rate</li>
 *   <li>{@link com.rts.jnn.core.decay.StepDecay} - Discrete drops at regular intervals</li>
 * </ul>
 *
 * <h2>Decay Function Characteristics:</h2>
 * <table>
 *   <tr>
 *     <th>Decay Type</th>
 *     <th>Behavior</th>
 *     <th>Best For</th>
 *   </tr>
 *   <tr>
 *     <td>Exponential</td>
 *     <td>Rapid initial decay</td>
 *     <td>Quick convergence needs</td>
 *   </tr>
 *   <tr>
 *     <td>Inverse Time</td>
 *     <td>Gradual decay</td>
 *     <td>Stable, long-term training</td>
 *   </tr>
 *   <tr>
 *     <td>Polynomial</td>
 *     <td>Controlled decay to end rate</td>
 *     <td>Fixed-duration training</td>
 *   </tr>
 *   <tr>
 *     <td>Step</td>
 *     <td>Discrete rate drops</td>
 *     <td>Scheduled rate changes</td>
 *   </tr>
 * </table>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Create a decay function
 * DecayFunction decay = new ExponentialDecay(0.1, 0.01);
 *
 * // Use in training loop
 * for (int epoch = 0; epoch < maxEpochs; epoch++) {
 *     double currentRate = decay.getLearningRate(epoch);
 *     network.setLearningRate(currentRate);
 *     // ... training code ...
 * }
 * }</pre>
 *
 * <h2>Selection Guidelines:</h2>
 * <p>Choose a decay function based on:</p>
 * <ul>
 *   <li>Training duration requirements</li>
 *   <li>Convergence speed needs</li>
 *   <li>Stability requirements</li>
 *   <li>Whether discrete or continuous decay is preferred</li>
 * </ul>
 *
 * <h2>Implementation Notes:</h2>
 * <ul>
 *   <li>All implementations are thread-safe</li>
 *   <li>All functions are stateless</li>
 *   <li>Parameters are validated in constructors</li>
 *   <li>Learning rates never go below zero</li>
 * </ul>
 *
 * @see com.rts.jnn.core.decay.DecayFunction
 * @see com.rts.jnn.core.network.NeuralNetwork
 */
package com.rts.jnn.core.decay;