package com.rts.jnn.core.network;

import com.rts.jnn.core.activation.ActivationFunction;
import com.rts.jnn.core.initialization.InitializationFunction;

/**
 * Represents a layer of neurons in the neural network.
 *
 * <p>A layer consists of multiple neurons, each using the same activation function
 * and initialization strategy. Each neuron connects to every neuron in the previous
 * layer (or to all inputs for the input layer).</p>
 *
 * <h2>Layer Properties:</h2>
 * <ul>
 *   <li>Fixed number of neurons</li>
 *   <li>Common activation function for all neurons</li>
 *   <li>Common initialization strategy</li>
 *   <li>Full connectivity with previous layer</li>
 * </ul>
 *
 * <h2>Example Usage:</h2>
 * <pre>{@code
 * // Create a layer with 10 neurons, 5 inputs
 * Layer layer = new Layer(
 *     10,                        // neuron count
 *     5,                         // input size
 *     new SigmoidActivation(),   // activation function
 *     new XavierInitialization() // weight initialization
 * );
 *
 * // Forward propagation
 * double[] inputs = {1.0, 0.5, -0.5, 0.8, -0.2};
 * double[] outputs = layer.forward(inputs);
 * }</pre>
 */
public class Layer {

    /**
     * Array of neurons in this layer
     */
    private Neuron[] neurons;

    /**
     * Output values from the last forward pass
     */
    private double[] outputs;

    /**
     * Activation function used by all neurons in the layer
     */
    private ActivationFunction activationFunction;

    /**
     * Weight initialization strategy for the layer
     */
    private InitializationFunction initializationFunction;

    /**
     * Creates a new layer with the specified configuration.
     *
     * @param neuronCount            Number of neurons in this layer
     * @param inputSize              Number of inputs to each neuron
     * @param activationFunction     Activation function for all neurons
     * @param initializationFunction Weight initialization strategy
     * @throws IllegalArgumentException if neuronCount or inputSize is less than 1
     */
    public Layer(
            int neuronCount,
            int inputSize,
            ActivationFunction activationFunction,
            InitializationFunction initializationFunction
    ) {
        neurons = new Neuron[neuronCount];
        this.activationFunction = activationFunction;
        this.initializationFunction = initializationFunction;

        // Initialize each neuron
        for (int i = 0; i < neuronCount; i++) {
            neurons[i] = new Neuron(
                    initializationFunction.init(inputSize),
                    0.0,
                    activationFunction
            );
        }
    }

    /**
     * Performs forward propagation through the layer.
     *
     * @param inputs Input vector for the layer
     * @return Output vector containing each neuron's activation
     */
    public double[] forward(double[] inputs) {
        outputs = new double[neurons.length];
        for (int i = 0; i < neurons.length; i++) {
            outputs[i] = neurons[i].activate(inputs);
        }
        return outputs;
    }

    // Getters and setters
    public Neuron[] getNeurons() {
        return neurons;
    }

    public void setNeurons(Neuron[] neurons) {
        this.neurons = neurons;
    }

    public double[] getOutputs() {
        return outputs;
    }

    public void setOutputs(double[] outputs) {
        this.outputs = outputs;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    public InitializationFunction getInitializationFunction() {
        return initializationFunction;
    }

    public void setInitializationFunction(InitializationFunction initializationFunction) {
        this.initializationFunction = initializationFunction;
    }
}