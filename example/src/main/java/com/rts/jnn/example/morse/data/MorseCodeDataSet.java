package com.rts.jnn.example.morse.data;

import com.rts.jnn.example.morse.util.Utils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Manages the training dataset for Morse code translation.
 *
 * <p>This class maintains the standard mapping between Morse code sequences and
 * their corresponding characters, and generates appropriate training vectors for
 * the neural network. It supports:</p>
 *
 * <ul>
 *   <li>All 26 letters (A-Z)</li>
 *   <li>All 10 digits (0-9)</li>
 *   <li>Space character</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * MorseCodeDataSet dataset = new MorseCodeDataSet();
 * for (int i = 0; i < dataset.inputs.size(); i++) {
 *     double[] input = dataset.inputs.get(i);
 *     double[] target = dataset.outputs.get(i);
 *     network.train(input, target);
 * }
 * }</pre>
 *
 * <p><b>Thread Safety:</b> This class is thread-safe as it's effectively immutable
 * after construction.</p>
 */
public class MorseCodeDataSet {

    /**
     * Maps Morse code sequences to their corresponding characters
     */
    public static final Map<String, String> MORSE_TO_LETTER = new HashMap<>();

    /**
     * Maps characters to their corresponding Morse code sequences
     */
    public static final Map<String, String> LETTER_TO_MORSE = new HashMap<>();

    static {
        String[][] codes = {
                {"A", ".-"}, {"B", "-..."}, {"C", "-.-."}, {"D", "-.."},
                {"E", "."}, {"F", "..-."}, {"G", "--."}, {"H", "...."},
                {"I", ".."}, {"J", ".---"}, {"K", "-.-"}, {"L", ".-.."},
                {"M", "--"}, {"N", "-."}, {"O", "---"}, {"P", ".--."},
                {"Q", "--.-"}, {"R", ".-."}, {"S", "..."}, {"T", "-"},
                {"U", "..-"}, {"V", "...-"}, {"W", ".--"}, {"X", "-..-"},
                {"Y", "-.--"}, {"Z", "--.."}, {"1", ".----"}, {"2", "..---"},
                {"3", "...--"}, {"4", "....-"}, {"5", "....."}, {"6", "-...."},
                {"7", "--..."}, {"8", "---.."}, {"9", "----."}, {"0", "-----"},
                {" ", "/"},     // Using '/' to represent space between words
        };

        for (String[] code : codes) {
            String letter = code[0];
            String morse = code[1];
            MORSE_TO_LETTER.put(morse, letter);
            LETTER_TO_MORSE.put(letter, morse);
        }
    }

    /**
     * Training input vectors
     */
    public List<double[]> inputs;

    /**
     * Corresponding target output vectors
     */
    public List<double[]> outputs;

    /**
     * Initializes the dataset with all Morse code mappings and
     * generates training vectors for the neural network.
     */
    public MorseCodeDataSet() {
        inputs = new ArrayList<>();
        outputs = new ArrayList<>();

        // Generate training data for each Morse code sequence
        for (Map.Entry<String, String> entry : MorseCodeDataSet.MORSE_TO_LETTER.entrySet()) {
            String morseCode = entry.getKey();
            String letter = entry.getValue();

            // Convert Morse code to input vector and letter to output vector
            double[] inputVector = Utils.encodeMorseCode(morseCode, 5);
            double[] outputVector = encodeLetter(letter);

            inputs.add(inputVector);
            outputs.add(outputVector);
        }
    }

    /**
     * Encodes a letter into a binary vector representation.
     * Creates a vector of size 36 (26 letters + 10 digits) with
     * a 1.0 at the position corresponding to the letter/digit.
     *
     * @param letter The character to encode
     * @return Binary vector representation
     */
    private double[] encodeLetter(String letter) {
        double[] vector = new double[36]; // 26 letters + 10 digits
        int index = getLetterIndex(letter);
        vector[index] = 1.0;
        return vector;
    }

    private int getLetterIndex(String letter) {
        if (letter.equals(" ")) {
            return 35; // Index for space
        } else if (Character.isDigit(letter.charAt(0))) {
            return 26 + (letter.charAt(0) - '0'); // Digits index from 26 to 35
        } else {
            return letter.charAt(0) - 'A'; // Letters index from 0 to 25
        }
    }
}
