package com.rts.jnn.core.network;

import com.rts.jnn.core.activation.ActivationFunction;
import com.rts.jnn.core.exception.NetworkConfigurationException;
import com.rts.jnn.core.exception.NeuralNetworkException;
import com.rts.jnn.core.exception.TrainingException;
import com.rts.jnn.core.initialization.InitializationFunction;
import com.rts.jnn.core.validation.ValidationUtils;

import java.util.ArrayList;
import java.util.List;

/**
 * Provides the main neural network implementation for Morse code translation.
 *
 * <p>This class implements a feed-forward neural network that can be trained to translate
 * Morse code sequences into alphanumeric characters. The network supports:</p>
 *
 * <ul>
 *   <li>Multiple hidden layers</li>
 *   <li>Configurable activation functions per layer</li>
 *   <li>Custom weight initialization strategies</li>
 *   <li>Dynamic learning rate adjustment</li>
 * </ul>
 *
 * <p><b>Thread Safety:</b> This class is not thread-safe and should not be accessed concurrently.</p>
 *
 * <h2>Usage Example:</h2>
 * <pre>{@code
 * // Create network
 * NeuralNetwork network = new NeuralNetwork(0.5);
 *
 * // Configure layers
 * network.addLayer(5, new SigmoidActivation(), new XavierInitialization());
 * network.addLayer(36, new SigmoidActivation(), new XavierInitialization());
 *
 * // Train network
 * double[] inputs = Utils.encodeMorseCode(".-", 5);
 * double[] targets = new double[36]; // One-hot encoded target
 * targets[0] = 1.0; // Target 'A'
 * network.train(inputs, targets);
 * }</pre>
 *
 * @see Layer
 * @see Neuron
 * @see com.rts.jnn.core.activation.ActivationFunction
 */
public class NeuralNetwork {
    private List<Layer> layers;
    private double learningRate;
    private double initialLearningRate;

    /**
     * Creates a new neural network with the specified learning rate.
     *
     * <p>Initializes an empty network with no layers. Layers must be added using
     * {@link #addLayer(int, ActivationFunction, InitializationFunction)}.</p>
     *
     * @param learningRate Initial learning rate in range (0,1], typically 0.1 or 0.01
     * @throws IllegalArgumentException if learning rate is not in range (0,1]
     */
    public NeuralNetwork(double learningRate) {
        if (learningRate <= 0 || learningRate > 1) {
            throw new IllegalArgumentException("Learning rate must be in range (0,1]");
        }
        this.learningRate = learningRate;
        this.initialLearningRate = learningRate;
        this.layers = new ArrayList<>();
    }

    public List<Layer> getLayers() {
        return layers;
    }

    public void setLayers(List<Layer> layers) {
        this.layers = layers;
    }

    public double getLearningRate() {
        return learningRate;
    }

    public void setLearningRate(double learningRate) {
        this.learningRate = learningRate;
    }

    public double getInitialLearningRate() {
        return initialLearningRate;
    }

    public void setInitialLearningRate(double initialLearningRate) {
        this.initialLearningRate = initialLearningRate;
    }

    /**
     * Adds a new layer to the neural network.
     *
     * <p>Layers are added sequentially, with each layer automatically connected to the
     * previous layer. The first layer added becomes the input layer, and the last
     * layer added becomes the output layer.</p>
     *
     * <p>The input size for the layer is automatically determined:</p>
     * <ul>
     *   <li>For the first layer: equal to neuronCount</li>
     *   <li>For subsequent layers: equal to previous layer's neuron count</li>
     * </ul>
     *
     * @param neuronCount            Number of neurons in the layer
     * @param activationFunction     Activation function for all neurons in the layer
     * @param initializationFunction Weight initialization strategy for the layer
     * @throws IllegalArgumentException if neuronCount is less than 1
     * @throws NullPointerException     if activationFunction or initializationFunction is null
     */
    public void addLayer(int neuronCount, ActivationFunction activationFunction, InitializationFunction initializationFunction) {
        // Validate parameters
        ValidationUtils.validateLayerConfig(neuronCount,
                layers.isEmpty() ? neuronCount : layers.get(layers.size() - 1).getNeurons().length,
                activationFunction, initializationFunction);

        int inputSize = layers.isEmpty() ? neuronCount : layers.get(layers.size() - 1).getNeurons().length;
        layers.add(new Layer(neuronCount, inputSize, activationFunction, initializationFunction));
    }

    /**
     * Performs forward propagation to generate predictions.
     *
     * <p>Takes an input vector and propagates it through the network to produce
     * an output vector. For Morse code translation, the output vector represents
     * probabilities for each possible character.</p>
     *
     * @param inputs Input vector matching the size of the first layer
     * @return Output vector with size matching the final layer
     * @throws IllegalArgumentException if inputs length doesn't match input layer size
     * @throws IllegalStateException    if network has no layers
     */
    public double[] predict(double[] inputs) {
        // Validate network state and inputs
        ValidationUtils.validateNetworkState(this);
        ValidationUtils.validateInputVector(inputs, layers.get(0).getNeurons()[0].getWeights().length);

        try {
            double[] outputs = inputs;
            for (Layer layer : layers) {
                outputs = layer.forward(outputs);
            }
            return outputs;
        } catch (Exception e) {
            throw new NeuralNetworkException("Error during prediction", e);
        }
    }

    /**
     * Trains the network using backpropagation.
     *
     * <p>Performs one training iteration:</p>
     * <ol>
     *   <li>Forward propagation to compute outputs</li>
     *   <li>Error calculation</li>
     *   <li>Backpropagation of errors</li>
     *   <li>Weight and bias updates</li>
     * </ol>
     *
     * @param inputs  Training input vector
     * @param targets Target output vector
     * @throws IllegalArgumentException if vector dimensions don't match network
     * @throws IllegalStateException    if network has no layers
     */
    public void train(double[] inputs, double[] targets) {
        // Validate network state and training data
        ValidationUtils.validateNetworkState(this);
        ValidationUtils.validateTrainingData(inputs, targets, this);

        try {
            double[] outputs = predict(inputs);

            // Back propagation
            double[] errors = new double[outputs.length];
            for (int i = 0; i < outputs.length; i++) {
                errors[i] = targets[i] - outputs[i];
            }

            for (int i = layers.size() - 1; i >= 0; i--) {
                Layer layer = layers.get(i);
                double[] nextErrors = new double[layer.getNeurons()[0].getWeights().length];

                for (int j = 0; j < layer.getNeurons().length; j++) {
                    Neuron neuron = layer.getNeurons()[j];

                    // Calculate delta with null checks
                    if (neuron.getActivationFunction() == null) {
                        throw new NetworkConfigurationException("Neuron activation function is null");
                    }

                    neuron.setDelta(errors[j] * neuron.getActivationFunction().derivative(neuron.getOutput()));

                    // Update weights and biases
                    double[] inputsToUse = i == 0 ? inputs : layers.get(i - 1).getOutputs();

                    if (inputsToUse == null) {
                        throw new TrainingException("Layer inputs are null during backpropagation");
                    }

                    for (int k = 0; k < neuron.getWeights().length; k++) {
                        neuron.getWeights()[k] += learningRate * neuron.getDelta() * inputsToUse[k];
                        nextErrors[k] += neuron.getWeights()[k] * neuron.getDelta();
                    }
                    neuron.setBias(neuron.getBias() + learningRate * neuron.getDelta());
                }
                errors = nextErrors;
            }
        } catch (Exception e) {
            throw new TrainingException("Error during training: " + e.getMessage());
        }
    }

}

