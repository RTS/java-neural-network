package com.rts.jnn.core.decay;

/**
 * Defines the contract for learning rate decay functions in neural network training.
 *
 * <p>Decay functions help optimize training by gradually reducing the learning rate
 * over time. This can improve both training stability and final model performance.
 * A proper decay schedule helps the network to:</p>
 *
 * <ul>
 *   <li>Make large learning steps early in training</li>
 *   <li>Take smaller, more refined steps as training progresses</li>
 *   <li>Avoid overshooting optimal values</li>
 *   <li>Fine-tune weights in later stages</li>
 * </ul>
 *
 * <h2>Available Implementations:</h2>
 * <ul>
 *   <li>{@link ExponentialDecay} - rate = initial * e^(-decay * epoch)</li>
 *   <li>{@link InverseTimeDecay} - rate = initial / (1 + decay * epoch)</li>
 *   <li>{@link PolynomialDecay} - rate = (initial - end) * (1 - epoch/maxEpochs)^power + end</li>
 *   <li>{@link StepDecay} - rate = initial * (decay ^ floor(epoch/steps))</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // Create a decay function
 * DecayFunction decay = new ExponentialDecay(0.1, 0.01);
 *
 * // In training loop
 * for (int epoch = 0; epoch < maxEpochs; epoch++) {
 *     double rate = decay.getLearningRate(epoch);
 *     network.setLearningRate(rate);
 *     network.train(inputs, targets);
 * }
 * }</pre>
 *
 * <h2>Implementation Guidelines:</h2>
 * <p>When implementing this interface:</p>
 * <ul>
 *   <li>Ensure thread-safety (implementations should be immutable)</li>
 *   <li>Validate constructor parameters</li>
 *   <li>Never return negative learning rates</li>
 *   <li>Handle edge cases (e.g., epoch = 0)</li>
 *   <li>Document the mathematical formula used</li>
 * </ul>
 *
 * @see com.rts.jnn.core.network.NeuralNetwork
 * @see com.rts.jnn.core.decay.ExponentialDecay
 * @see com.rts.jnn.core.decay.InverseTimeDecay
 * @see com.rts.jnn.core.decay.PolynomialDecay
 * @see com.rts.jnn.core.decay.StepDecay
 */
public interface DecayFunction {

    /**
     * Calculates the learning rate for a given epoch.
     *
     * <p>This method should return a positive learning rate value that typically
     * decreases as the epoch number increases. The specific decay pattern depends
     * on the implementation.</p>
     *
     * <p>Implementations should ensure:</p>
     * <ul>
     *   <li>Return value is always positive</li>
     *   <li>Consistent results for same epoch value</li>
     *   <li>Smooth or step-wise decay as appropriate</li>
     *   <li>Proper handling of edge cases</li>
     * </ul>
     *
     * @param epoch Current training epoch (non-negative)
     * @return Adjusted learning rate for the epoch
     * @throws IllegalArgumentException if epoch is negative
     */
    double getLearningRate(int epoch);
}