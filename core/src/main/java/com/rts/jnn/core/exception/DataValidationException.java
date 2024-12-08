package com.rts.jnn.core.exception;

/**
 * Exception thrown for data validation errors.
 */
public class DataValidationException extends NeuralNetworkException {
    public DataValidationException(String message) {
        super(message);
    }
}