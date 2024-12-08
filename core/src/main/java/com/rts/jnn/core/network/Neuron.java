package com.rts.jnn.core.network;

import com.rts.jnn.core.activation.ActivationFunction;

/**
 * Implements an individual neuron in the neural network.
 *
 * <p>Each neuron:</p>
 * <ul>
 *   <li>Maintains a set of weights for its inputs</li>
 *   <li>Has an adjustable bias term</li>
 *   <li>Uses an activation function to compute its output</li>
 *   <li>Stores state needed for backpropagation</li>
 * </ul>
 *
 * <p>The neuron's output is computed as: activation(sum(weights * inputs) + bias)</p>
 *
 * <p><b>Thread Safety:</b> This class is not thread-safe.</p>
 */
public class Neuron {

    private double[] weights;
    private double bias;
    private double output;
    private double delta;
    private ActivationFunction activationFunction;

    /**
     * Creates a new neuron with specified weights, bias and activation function.
     *
     * <p>The number of weights must match the number of inputs the neuron will receive.</p>
     *
     * @param weights            Initial weight values for input connections
     * @param bias               Initial bias value
     * @param activationFunction Function to transform weighted sum to output
     * @throws NullPointerException if weights or activationFunction is null
     */
    public Neuron(double[] weights, double bias, ActivationFunction activationFunction) {
        this.weights = weights;
        this.bias = bias;
        this.activationFunction = activationFunction;
    }

    public double[] getWeights() {
        return weights;
    }

    public void setWeights(double[] weights) {
        this.weights = weights;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double getOutput() {
        return output;
    }

    public void setOutput(double output) {
        this.output = output;
    }

    public double getDelta() {
        return delta;
    }

    public void setDelta(double delta) {
        this.delta = delta;
    }

    public ActivationFunction getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(ActivationFunction activationFunction) {
        this.activationFunction = activationFunction;
    }

    /**
     * Computes the neuron's output for given inputs.
     *
     * <p>The computation follows these steps:</p>
     * <ol>
     *   <li>Calculate weighted sum of inputs plus bias</li>
     *   <li>Apply activation function</li>
     *   <li>Store and return result</li>
     * </ol>
     *
     * @param inputs Input values, must match number of weights
     * @return Neuron's output value
     * @throws IllegalArgumentException if inputs length doesn't match weights length
     */
    public double activate(double[] inputs) {
        double sum = bias;
        for (int i = 0; i < weights.length; i++) {
            sum += weights[i] * inputs[i];
        }
        output = activationFunction.activate(sum);
        return output;
    }

}
