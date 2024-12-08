package com.rts.jnn.core.decay;

/**
 * Implements polynomial decay for learning rate adjustment during training.
 *
 * <p>The learning rate decays according to the formula:</p>
 * <pre>learningRate = (initialRate - endRate) * (1 - epoch/maxEpochs)^power + endRate</pre>
 *
 * <p>This decay schedule provides:</p>
 * <ul>
 *   <li>Controlled decay to a specified end rate</li>
 *   <li>Flexible decay curve based on power parameter</li>
 *   <li>Guaranteed minimum learning rate</li>
 *   <li>Predictable training duration</li>
 * </ul>
 *
 * <h2>Mathematical Properties:</h2>
 * <ul>
 *   <li>Monotonically decreasing</li>
 *   <li>Reaches exactly endRate at maxEpochs</li>
 *   <li>Decay speed controlled by power parameter</li>
 *   <li>Power > 1: Slower initial decay</li>
 *   <li>Power < 1: Faster initial decay</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // Create decay with:
 * // - Initial rate: 0.1
 * // - End rate: 0.001
 * // - Max epochs: 1000
 * // - Power: 2.0
 * DecayFunction decay = new PolynomialDecay(0.1, 0.001, 1000, 2.0);
 *
 * // Get learning rates for different epochs
 * double rate0 = decay.getLearningRate(0);     // = 0.1000
 * double rate500 = decay.getLearningRate(500); // ≈ 0.0260
 * double rate1000 = decay.getLearningRate(1000); // = 0.001
 * }</pre>
 */
public record PolynomialDecay(double initialLearningRate, double endLearningRate, int maxEpochs,
                              double power) implements DecayFunction {
    /**
     * Creates a new polynomial decay function.
     *
     * @param initialLearningRate Starting learning rate value
     * @param endLearningRate     Final learning rate value
     * @param maxEpochs           Number of epochs for complete decay
     * @param power               Power of the polynomial decay (controls decay curve)
     * @throws IllegalArgumentException if parameters are invalid
     */
    public PolynomialDecay {
        if (initialLearningRate < 0) {
            throw new IllegalArgumentException("Initial learning rate must be non-negative");
        }
        if (endLearningRate < 0) {
            throw new IllegalArgumentException("End learning rate must be non-negative");
        }
        if (endLearningRate > initialLearningRate) {
            throw new IllegalArgumentException("End learning rate must be less than initial rate");
        }
        if (maxEpochs <= 0) {
            throw new IllegalArgumentException("Max epochs must be positive");
        }
        if (power <= 0) {
            throw new IllegalArgumentException("Power must be positive");
        }

    }

    /**
     * Calculates the learning rate for a given epoch.
     *
     * <p>The returned learning rate follows a polynomial decay curve, ensuring
     * a smooth transition from initial to end learning rate.</p>
     *
     * @param epoch Current training epoch (non-negative)
     * @return Adjusted learning rate for the epoch
     */
    @Override
    public double getLearningRate(int epoch) {
        if (epoch >= maxEpochs) {
            return endLearningRate;
        }

        double decayProgress = 1.0 - (double) epoch / maxEpochs;
        double decayFactor = Math.pow(decayProgress, power);
        return (initialLearningRate - endLearningRate) * decayFactor + endLearningRate;
    }
}