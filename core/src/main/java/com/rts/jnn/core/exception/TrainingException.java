package com.rts.jnn.core.exception;

/**
 * Exception thrown for training related errors.
 */
public class TrainingException extends NeuralNetworkException {

    public TrainingException(String message) {
        super(message);
    }
}