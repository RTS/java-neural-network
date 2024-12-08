package com.rts.jnn.example.morse.util;

/**
 * Utility functions for neural network operations.
 *
 * <p>This class provides various helper methods for:</p>
 * <ul>
 *   <li>Morse code encoding/decoding</li>
 *   <li>Vector operations</li>
 *   <li>Activation function management</li>
 *   <li>Initialization function management</li>
 * </ul>
 */
public class Utils {

    /**
     * Encodes a Morse code sequence into a fixed-length vector.
     *
     * <p>Encoding scheme:</p>
     * <ul>
     *   <li>Dot (.): 1.0</li>
     *   <li>Dash (-): -1.0</li>
     *   <li>Padding: 0.0</li>
     * </ul>
     *
     * @param morseCode Morse code sequence to encode
     * @param maxLength Maximum length of the vector
     * @return Encoded vector representation
     * @throws IllegalArgumentException if maxLength is less than 1
     */
    public static double[] encodeMorseCode(String morseCode, int maxLength) {
        double[] vector = new double[maxLength];

        for (int i = 0; i < maxLength; i++) {
            if (i < morseCode.length()) {
                char c = morseCode.charAt(i);
                if (c == '.') {
                    vector[i] = 1.0;
                } else if (c == '-') {
                    vector[i] = -1.0;
                } else {
                    vector[i] = 0.0;
                }
            } else {
                vector[i] = 0.0; // Padding
            }
        }
        return vector;
    }

    /**
     * Decodes a network output vector into the corresponding character.
     *
     * <p>Selects the index with highest activation value and maps it to
     * the corresponding character (A-Z, 0-9, or space).</p>
     *
     * @param outputVector Network output vector
     * @return Decoded character
     * @throws IllegalArgumentException if outputVector is null or empty
     */
    public static String decodeOutput(double[] outputVector) {
        int index = -1;
        double max = -Double.MAX_VALUE;
        for (int i = 0; i < outputVector.length; i++) {
            if (outputVector[i] > max) {
                max = outputVector[i];
                index = i;
            }
        }

        return getLetterFromIndex(index);
    }

    public static String getLetterFromIndex(int index) {
        if (index >= 0 && index < 26) {
            return String.valueOf((char) ('A' + index));
        } else if (index >= 26 && index < 36) {
            return String.valueOf((char) ('0' + index - 26));
        } else if (index == 35) {
            return " ";
        } else {
            return "?";
        }
    }

}
