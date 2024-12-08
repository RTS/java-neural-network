package com.rts.jnn.core.decay;

/**
 * Implements exponential decay for learning rate adjustment during training.
 *
 * <p>The learning rate decays exponentially according to the formula:</p>
 * <pre>learningRate = initialLearningRate * e^(-decayRate * epoch)</pre>
 *
 * <p>This decay schedule provides:</p>
 * <ul>
 *   <li>Rapid initial decrease in learning rate</li>
 *   <li>Gradual slowing of decay over time</li>
 *   <li>Smooth, continuous decay curve</li>
 *   <li>Never reaches exactly zero</li>
 * </ul>
 *
 * <h2>Mathematical Properties:</h2>
 * <ul>
 *   <li>Monotonically decreasing</li>
 *   <li>Asymptotically approaches zero</li>
 *   <li>Steepest descent at start</li>
 *   <li>Decay controlled by decay rate parameter</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // Create decay with initial rate 0.1 and decay rate 0.01
 * DecayFunction decay = new ExponentialDecay(0.1, 0.01);
 *
 * // Get learning rates for different epochs
 * double rate0 = decay.getLearningRate(0);    // = 0.1000
 * double rate10 = decay.getLearningRate(10);  // ≈ 0.0905
 * double rate100 = decay.getLearningRate(100); // ≈ 0.0367
 * }</pre>
 */
public record ExponentialDecay(double initialLearningRate, double decayRate) implements DecayFunction {
    /**
     * Creates a new exponential decay function.
     *
     * @param initialLearningRate Starting learning rate value
     * @param decayRate           Controls how quickly the rate decays (larger = faster decay)
     * @throws IllegalArgumentException if initialLearningRate or decayRate is negative
     */
    public ExponentialDecay {
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
     * <p>The returned learning rate follows an exponential decay curve:
     * rate = initialRate * e^(-decayRate * epoch)</p>
     *
     * @param epoch Current training epoch (non-negative)
     * @return Adjusted learning rate for the epoch
     */
    @Override
    public double getLearningRate(int epoch) {
        return initialLearningRate * Math.exp(-decayRate * epoch);
    }

    /**
     * Gets the initial learning rate.
     *
     * @return The initial learning rate value
     */
    @Override
    public double initialLearningRate() {
        return initialLearningRate;
    }

    /**
     * Gets the decay rate.
     *
     * @return The decay rate value
     */
    @Override
    public double decayRate() {
        return decayRate;
    }
}