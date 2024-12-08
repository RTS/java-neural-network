package com.rts.jnn.core.decay;

/**
 * Implements inverse time decay for learning rate adjustment during training.
 *
 * <p>The learning rate decays according to the formula:</p>
 * <pre>learningRate = initialLearningRate / (1 + decayRate * epoch)</pre>
 *
 * <p>This decay schedule provides:</p>
 * <ul>
 *   <li>Slower decay compared to exponential</li>
 *   <li>Gradual reduction in learning rate</li>
 *   <li>Inverse proportional relationship with time</li>
 *   <li>More stable long-term training</li>
 * </ul>
 *
 * <h2>Mathematical Properties:</h2>
 * <ul>
 *   <li>Monotonically decreasing</li>
 *   <li>Asymptotically approaches zero</li>
 *   <li>Decay speed proportional to 1/t</li>
 *   <li>Smoother transition than exponential decay</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // Create decay with initial rate 0.1 and decay rate 0.01
 * DecayFunction decay = new InverseTimeDecay(0.1, 0.01);
 *
 * // Get learning rates for different epochs
 * double rate0 = decay.getLearningRate(0);    // = 0.1000
 * double rate10 = decay.getLearningRate(10);  // ≈ 0.0500
 * double rate100 = decay.getLearningRate(100); // ≈ 0.0091
 *
 * // Use in training loop
 * for (int epoch = 0; epoch < maxEpochs; epoch++) {
 *     double currentRate = decay.getLearningRate(epoch);
 *     network.setLearningRate(currentRate);
 *     // ... training code ...
 * }
 * }</pre>
 */
public record InverseTimeDecay(double initialLearningRate, double decayRate) implements DecayFunction {
    /**
     * Creates a new inverse time decay function.
     *
     * @param initialLearningRate Starting learning rate value
     * @param decayRate           Controls how quickly the rate decays (larger = faster decay)
     * @throws IllegalArgumentException if initialLearningRate is negative or decayRate is negative
     */
    public InverseTimeDecay {
        if (initialLearningRate < 0) {
            throw new IllegalArgumentException("Initial learning rate must be non-negative");
        }
        if (decayRate < 0) {
            throw new IllegalArgumentException("Decay rate must be non-negative");
        }
    }

    /**
     * Calculates the learning rate for a given epoch.
     *
     * <p>The returned learning rate follows an inverse time decay curve:
     * rate = initialRate / (1 + decayRate * epoch)</p>
     *
     * @param epoch Current training epoch (non-negative)
     * @return Adjusted learning rate for the epoch
     */
    @Override
    public double getLearningRate(int epoch) {
        return initialLearningRate / (1 + decayRate * epoch);
    }
}