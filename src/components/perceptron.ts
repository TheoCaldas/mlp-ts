import { dotProduct } from './math';
import { ActivationFunction } from './activation';

// The perceptron has inputs, weights, bias, output and activation function
export type Perceptron = {
    inputs?: number[],
    weights: number[],
    bias: number,
    output?: number,
    activationFunc: ActivationFunction,
}

// Initialize a perceptron with given weights, bias and activation function
// Inputs and output are undefined at this point
export const init = (weights: number[], bias: number, func: ActivationFunction): Perceptron =>  {
    const perceptron = {
        weights: weights,
        bias: bias,
        activationFunc: func,
    };
    return perceptron;
}

// Initialize a perceptron with random weights and bias, and activation function
// Inputs and output are undefined at this point
export const initRandom = (nInputs: number, func: ActivationFunction): Perceptron =>  {
    const weights = Array.from({ length: nInputs }, () => Math.random());
    const bias = Math.random();
    return init(weights, bias, func);
}

// Update the perceptron output based on given inputs and current weights
// The output is calculated as the dot product of inputs and weights plus the bias
export const updateOutput = (perceptron: Perceptron, inputs: number[]): Perceptron => {
    if (inputs.length !== perceptron.weights.length) {
        throw new Error("Input length must match weights length");
    }
    perceptron.inputs = inputs;
    const zValue = dotProduct(perceptron.inputs, perceptron.weights) + perceptron.bias;
    perceptron.output = perceptron.activationFunc(zValue);

    return perceptron;
}