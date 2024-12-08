package com.rts.jnn.example.morse.data;

/**
 * Manages character encoding/decoding for the Morse code neural network.
 * Provides a consistent 37-neuron mapping scheme for:
 * - 26 letters (A-Z): indices 0-25
 * - 10 digits (0-9): indices 26-35
 * - Space: index 36
 */
public class CharacterEncoder {

    // Constants for the encoding scheme
    public static final int LETTER_COUNT = 26;
    public static final int DIGIT_COUNT = 10;
    public static final int TOTAL_NEURONS = LETTER_COUNT + DIGIT_COUNT + 1; // +1 for space

    /**
     * Encodes a character into its corresponding neuron index.
     *
     * @param character The character to encode
     * @return Index in the range [0, 36]
     * @throws IllegalArgumentException if character is invalid
     */
    public static int encodeCharacter(String character) {
        if (character == null || character.length() != 1) {
            throw new IllegalArgumentException("Input must be a single character");
        }

        char c = character.charAt(0);

        if (character.equals(" ")) {
            return LETTER_COUNT + DIGIT_COUNT; // Space gets the last index (36)
        } else if (Character.isUpperCase(c)) {
            return c - 'A'; // A=0, B=1, ..., Z=25
        } else if (Character.isLowerCase(c)) {
            return c - 'a'; // Convert to same indices as uppercase
        } else if (Character.isDigit(c)) {
            return LETTER_COUNT + (c - '0'); // 0=26, 1=27, ..., 9=35
        }

        throw new IllegalArgumentException("Invalid character: " + character);
    }

    /**
     * Decodes a neuron index back to its corresponding character.
     *
     * @param index The neuron index to decode
     * @return The corresponding character
     * @throws IllegalArgumentException if index is out of range
     */
    public static String decodeIndex(int index) {
        if (index < 0 || index >= TOTAL_NEURONS) {
            throw new IllegalArgumentException("Index out of range: " + index);
        }

        if (index < LETTER_COUNT) {
            return String.valueOf((char) ('A' + index));
        } else if (index < LETTER_COUNT + DIGIT_COUNT) {
            return String.valueOf(index - LETTER_COUNT);
        } else {
            return " ";
        }
    }

    /**
     * Creates a one-hot encoded vector for a character.
     *
     * @param character The character to encode
     * @return A vector of length TOTAL_NEURONS with a single 1.0 value
     */
    public static double[] createOutputVector(String character) {
        double[] vector = new double[TOTAL_NEURONS];
        vector[encodeCharacter(character)] = 1.0;
        return vector;
    }

    /**
     * Decodes a network output vector into a character.
     *
     * @param outputVector The network output to decode
     * @return The most likely character
     */
    public static String decodeOutputVector(double[] outputVector) {
        if (outputVector == null || outputVector.length != TOTAL_NEURONS) {
            throw new IllegalArgumentException("Invalid output vector length");
        }

        int maxIndex = 0;
        double maxValue = outputVector[0];

        for (int i = 1; i < outputVector.length; i++) {
            if (outputVector[i] > maxValue) {
                maxValue = outputVector[i];
                maxIndex = i;
            }
        }

        return decodeIndex(maxIndex);
    }
}