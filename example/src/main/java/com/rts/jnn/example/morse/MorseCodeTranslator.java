package com.rts.jnn.example.morse;

import com.rts.jnn.core.activation.SigmoidActivation;
import com.rts.jnn.core.activation.SwishActivation;
import com.rts.jnn.core.decay.DecayFunction;
import com.rts.jnn.core.decay.InverseTimeDecay;
import com.rts.jnn.core.initialization.XavierInitialization;
import com.rts.jnn.core.network.NeuralNetwork;
import com.rts.jnn.example.morse.data.MorseCodeDataSet;
import com.rts.jnn.example.morse.util.Utils;
import com.rts.jnn.persistence.ModelPersistence;

import java.io.File;

/**
 * Example usage of the neural network for Morse code translation.
 * <p>
 * This class demonstrates:
 * 1. Network configuration
 * 2. Training process
 * 3. Making predictions
 * 4. Model persistence
 */
public class MorseCodeTranslator {

    public static void main(String[] args) {

        // Initialize the dataset
        MorseCodeDataSet dataSet = new MorseCodeDataSet();

        // Create and configure the neural network
        com.rts.jnn.core.network.NeuralNetwork nn = new com.rts.jnn.core.network.NeuralNetwork(0.5); // Learning rate set to 0.5

        // Configure the neural network layers
        nn.addLayer(5, new SwishActivation(), new XavierInitialization()); // Input layer with 5 neurons
        nn.addLayer(20, new SigmoidActivation(), new XavierInitialization()); // Hidden layer with 20 neurons
        nn.addLayer(36, new SigmoidActivation(), new XavierInitialization()); // Output layer with 36 neurons

        // Generate model file name based on the network structure
        String modelFileName = ModelPersistence.generateModelFileName(nn);

        // Check if model file exists
        File modelFile = new File(modelFileName);
        if (modelFile.exists()) {
            // Load the model
            ModelPersistence.loadModel(modelFileName, nn);
        } else {

            // Train the neural network
            int epochs = 10_000;
            train(epochs, nn, new InverseTimeDecay(0.5, 0.0001), dataSet);

            // Save the model after training
            ModelPersistence.saveModel(modelFileName, nn);
        }

        // Test the neural network with sample Morse codes
        String[] testMorseCodes = {"-", "....", "..", "...", "..", "...", ".-", "-", ".", "...", "-"};
        test(testMorseCodes, nn);
    }

    private static void test(String[] testMorseCodes, NeuralNetwork nn) {
        StringBuilder message = new StringBuilder();

        for (String morseCode : testMorseCodes) {
            double[] inputVector = Utils.encodeMorseCode(morseCode, 5);
            double[] outputVector = nn.predict(inputVector);
            String predictedLetter = Utils.decodeOutput(outputVector);

            message.append(predictedLetter);
            System.out.println("Morse Code: " + morseCode + " -> Predicted Letter: " + predictedLetter);
        }
        System.out.println("Decoded Message: " + message);
    }

    private static void train(int epochs, NeuralNetwork nn, DecayFunction decayFunction, MorseCodeDataSet dataSet) {
        for (int epoch = 0; epoch < epochs; epoch++) {

            // learning rate schedule
            nn.setLearningRate(decayFunction.getLearningRate(epoch));

            double totalError = 0.0;
            for (int i = 0; i < dataSet.inputs.size(); i++) {
                double[] inputs = dataSet.inputs.get(i);
                double[] targets = dataSet.outputs.get(i);

                nn.train(inputs, targets);

                double[] outputs = nn.predict(inputs);
                for (int j = 0; j < targets.length; j++) {
                    double error = targets[j] - outputs[j];
                    totalError += error * error;
                }
            }

            if (epoch % 1000 == 0) {
                System.out.println("Epoch " + epoch + ", Error: " + totalError);
            }
        }
    }

}
