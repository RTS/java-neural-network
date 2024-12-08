package com.rts.jnn.core.decay;

/**
 * Implements step decay for learning rate adjustment during training.
 *
 * <p>The learning rate is reduced by a factor after a specified number of epochs.
 * Learning rate follows the formula:</p>
 * <pre>learningRate = initialRate * (decayFactor ^ floor(epoch/dropEvery))</pre>
 *
 * <p>For example, with:</p>
 * <ul>
 *   <li>Initial rate: 0.1</li>
 *   <li>Decay factor: 0.5</li>
 *   <li>Drop every: 10 epochs</li>
 * </ul>
 * <p>Results in:</p>
 * <ul>
 *   <li>Epochs 0-9: 0.1</li>
 *   <li>Epochs 10-19: 0.05</li>
 *   <li>Epochs 20-29: 0.025</li>
 *   <li>And so on...</li>
 * </ul>
 *
 * <h2>Mathematical Properties:</h2>
 * <ul>
 *   <li>Discrete drops in learning rate</li>
 *   <li>Constant rate between drops</li>
 *   <li>Geometric progression of rates</li>
 *   <li>Predictable schedule</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // Create decay with initial rate 0.1, decay factor 0.5, drop every 10 epochs
 * DecayFunction decay = new StepDecay(0.1, 0.5, 10);
 *
 * // Get learning rates for different epochs
 * double rate5 = decay.getLearningRate(5);   // = 0.1000
 * double rate15 = decay.getLearningRate(15); // = 0.0500
 * double rate25 = decay.getLearningRate(25); // = 0.0250
 *
 * // Use in training loop
 * for (int epoch = 0; epoch < maxEpochs; epoch++) {
 *     double currentRate = decay.getLearningRate(epoch);
 *     network.setLearningRate(currentRate);
 *     // ... training code ...
 * }
 * }</pre>
 */
public record StepDecay(double initialLearningRate, double decayFactor, int dropEvery) implements DecayFunction {
    /**
     * Creates a new step decay function.
     *
     * @param initialLearningRate Starting learning rate value
     * @param decayFactor         Factor to multiply rate by at each step (e.g., 0.5)
     * @param dropEvery           Number of epochs between rate updates
     * @throws IllegalArgumentException if parameters are invalid
     */
    public StepDecay {
        if (initialLearningRate <= 0) {
            throw new IllegalArgumentException("Initial learning rate must be positive");
        }
        if (decayFactor <= 0 || decayFactor >= 1) {
            throw new IllegalArgumentException("Decay factor must be between 0 and 1");
        }
        if (dropEvery <= 0) {
            throw new IllegalArgumentException("Drop interval must be positive");
        }

    }

    /**
     * Calculates the learning rate for a given epoch.
     *
     * <p>The rate drops by the decay factor every 'dropEvery' epochs.</p>
     *
     * @param epoch Current training epoch (non-negative)
     * @return Current learning rate
     */
    @Override
    public double getLearningRate(int epoch) {
        return initialLearningRate * Math.pow(decayFactor, (double) epoch / dropEvery);
    }
}