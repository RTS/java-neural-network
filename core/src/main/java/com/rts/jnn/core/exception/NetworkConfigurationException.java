package com.rts.jnn.core.exception;

/**
 * Exception thrown for invalid network configurations.
 */
public class NetworkConfigurationException extends NeuralNetworkException {
    public NetworkConfigurationException(String message) {
        super(message);
    }
}