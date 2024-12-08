package com.rts.jnn.persistence;

import com.rts.jnn.core.activation.ActivationFunction;
import com.rts.jnn.core.initialization.InitializationFunction;
import com.rts.jnn.core.network.Layer;
import com.rts.jnn.core.network.NeuralNetwork;
import com.rts.jnn.core.network.Neuron;
import com.rts.jnn.utils.Utils;

import java.io.*;

/**
 * Provides model persistence functionality for neural networks.
 *
 * <p>This class handles saving and loading of trained neural networks,
 * allowing models to be:</p>
 * <ul>
 *   <li>Saved to disk after training</li>
 *   <li>Loaded for later use</li>
 *   <li>Shared between different applications</li>
 * </ul>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Save model
 * String fileName = ModelPersistence.generateModelFileName(network);
 * ModelPersistence.saveModel(fileName, network);
 *
 * // Load model
 * ModelPersistence.loadModel(fileName, network);
 * }</pre>
 */
public class ModelPersistence {

    /**
     * Generates a unique filename for the model based on its architecture.
     *
     * <p>The filename format is: model_[layer-sizes]_[learning-rate].txt</p>
     * <p>Example: model_5-20-36_0.5.txt</p>
     *
     * @param neuralNetwork Network to generate filename for
     * @return Generated filename
     * @throws NullPointerException if neuralNetwork is null
     */
    public static String generateModelFileName(NeuralNetwork neuralNetwork) {
        StringBuilder sb = new StringBuilder();
        sb.append("model_");
        for (Layer layer : neuralNetwork.getLayers()) {
            sb.append(layer.getNeurons().length).append("-");
        }
        sb.deleteCharAt(sb.length() - 1); // Remove trailing '-'
        sb.append("_");
        sb.append(neuralNetwork.getInitialLearningRate());
        sb.append(".txt");
        return sb.toString();
    }

    /**
     * Saves the neural network model to a file.
     *
     * <p>The file format includes:</p>
     * <ul>
     *   <li>Learning rate</li>
     *   <li>Layer configurations</li>
     *   <li>Neuron weights and biases</li>
     * </ul>
     *
     * @param fileName      Name of file to save to
     * @param neuralNetwork Network to save
     */
    public static void saveModel(String fileName, NeuralNetwork neuralNetwork) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            // Write learning rate
            writer.write("LearningRate: " + neuralNetwork.getInitialLearningRate());
            writer.newLine();
            writer.newLine();

            // Write details of each layer
            for (int i = 0; i < neuralNetwork.getLayers().size(); i++) {
                Layer layer = neuralNetwork.getLayers().get(i);
                writer.write("InitLayer " + i + " - Size: " + layer.getNeurons().length +
                        ", Activation: " + layer.getActivationFunction().getClass().getSimpleName() +
                        ", Init: " + layer.getInitializationFunction().getClass().getSimpleName());
                writer.newLine();
            }

            writer.newLine();
            for (int i = 0; i < neuralNetwork.getLayers().size(); i++) {
                Layer layer = neuralNetwork.getLayers().get(i);
                writer.write("Layer " + i);
                writer.newLine();
                for (int j = 0; j < layer.getNeurons().length; j++) {
                    Neuron neuron = layer.getNeurons()[j];
                    writer.write("Neuron " + j);
                    writer.newLine();
                    writer.write("Weights:");
                    for (double weight : neuron.getWeights()) {
                        writer.write(weight + ",");
                    }
                    writer.newLine();
                    writer.write("Bias:" + neuron.getBias());
                    writer.newLine();
                }
                writer.newLine();
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Current working directory: " + System.getProperty("user.dir"));
        System.out.println("Model saved to " + fileName);
    }

    public static void loadModel(String fileName, NeuralNetwork neuralNetwork) {
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            neuralNetwork.getLayers().clear(); // Clear any existing layers
            String line;

            // Read the learning rate
            line = reader.readLine();
            if (line != null && line.startsWith("LearningRate:")) {
                neuralNetwork.setLearningRate(Double.parseDouble(line.split(":")[1].trim()));
            }

            // Skip empty line
            reader.readLine();

            // Read layer initialization information
            while ((line = reader.readLine()) != null && line.startsWith("InitLayer")) {
                String[] parts = line.split(",");

                // Parse layer index and size
                String[] layerInfo = parts[0].split("-");
                int layerSize = Integer.parseInt(layerInfo[1].trim().split(":")[1].trim());

                // Parse activation function
                String activationName = parts[1].trim().split(":")[1].trim();
                ActivationFunction activationFunction = Utils.getActivationFunctionByName(activationName);

                // Parse initialization function
                String initName = parts[2].trim().split(":")[1].trim();
                InitializationFunction initializationFunction = Utils.getInitializationFunctionByName(initName);

                // Determine input size based on previous layer (or default to size for first layer)
                int inputSize = neuralNetwork.getLayers().isEmpty() ? layerSize :
                        neuralNetwork.getLayers().get(neuralNetwork.getLayers().size() - 1).getNeurons().length;

                // Create and add layer to the network
                Layer layer = new Layer(layerSize, inputSize, activationFunction, initializationFunction);
                neuralNetwork.getLayers().add(layer);
            }

            // Now we read the weights and biases of each neuron
            int currentLayer = -1;
            Layer layer = null;
            while ((line = reader.readLine()) != null) {
                if (line.startsWith("Layer")) {
                    // New layer, increment layer index
                    currentLayer++;
                    layer = neuralNetwork.getLayers().get(currentLayer);
                } else if (line.startsWith("Neuron")) {
                    // Read neuron weights
                    int neuronIndex = Integer.parseInt(line.split(" ")[1]);
                    Neuron neuron = layer.getNeurons()[neuronIndex];

                    // Read weights
                    line = reader.readLine();
                    if (line != null && line.startsWith("Weights:")) {
                        String[] weightStrings = line.split(":")[1].split(",");
                        for (int i = 0; i < neuron.getWeights().length; i++) {
                            neuron.getWeights()[i] = Double.parseDouble(weightStrings[i]);
                        }
                    }

                    // Read bias
                    line = reader.readLine();
                    if (line != null && line.startsWith("Bias:")) {
                        neuron.setBias(Double.parseDouble(line.split(":")[1]));
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        System.out.println("Model loaded from " + fileName);
    }
}
