package com.rts.jnn.core.exception;

/**
 * Base exception class for all neural network related exceptions.
 */
public class NeuralNetworkException extends RuntimeException {

    public NeuralNetworkException(String message) {
        super(message);
    }

    public NeuralNetworkException(String message, Throwable cause) {
        super(message, cause);
    }
}