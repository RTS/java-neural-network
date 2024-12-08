package com.rts.jnn.core.validation;

import com.rts.jnn.core.activation.ActivationFunction;
import com.rts.jnn.core.exception.DataValidationException;
import com.rts.jnn.core.exception.NetworkConfigurationException;
import com.rts.jnn.core.initialization.InitializationFunction;
import com.rts.jnn.core.network.NeuralNetwork;

/**
 * Provides validation utilities for neural network operations.
 */
public class ValidationUtils {

    /**
     * Validates input vector dimensions.
     */
    public static void validateInputVector(double[] vector, int expectedSize) {
        if (vector == null) {
            throw new DataValidationException("Input vector cannot be null");
        }
        if (vector.length != expectedSize) {
            throw new DataValidationException(
                    String.format("Input vector size mismatch. Expected: %d, Got: %d",
                            expectedSize, vector.length));
        }
    }

    /**
     * Validates layer configuration parameters.
     */
    public static void validateLayerConfig(int neuronCount, int inputSize, ActivationFunction activation, InitializationFunction init) {
        if (neuronCount <= 0) {
            throw new NetworkConfigurationException(
                    "Neuron count must be positive, got: " + neuronCount);
        }
        if (inputSize <= 0) {
            throw new NetworkConfigurationException(
                    "Input size must be positive, got: " + inputSize);
        }
        if (activation == null) {
            throw new NetworkConfigurationException(
                    "Activation function cannot be null");
        }
        if (init == null) {
            throw new NetworkConfigurationException(
                    "Initialization function cannot be null");
        }
    }

    /**
     * Validates network state for operations.
     */
    public static void validateNetworkState(NeuralNetwork network) {
        if (network.getLayers().isEmpty()) {
            throw new NetworkConfigurationException(
                    "Neural network has no layers configured");
        }
    }

    /**
     * Validates training data.
     */
    public static void validateTrainingData(double[] inputs, double[] targets, NeuralNetwork network) {
        if (inputs == null || targets == null) {
            throw new DataValidationException("Training data cannot be null");
        }

        int inputSize = network.getLayers().get(0).getNeurons()[0].getWeights().length;
        int outputSize = network.getLayers().get(network.getLayers().size() - 1)
                .getNeurons().length;

        if (inputs.length != inputSize) {
            throw new DataValidationException(
                    String.format("Input size mismatch. Expected: %d, Got: %d",
                            inputSize, inputs.length));
        }
        if (targets.length != outputSize) {
            throw new DataValidationException(
                    String.format("Target size mismatch. Expected: %d, Got: %d",
                            outputSize, targets.length));
        }
    }
}