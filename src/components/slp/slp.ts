import { dotProduct } from '../math';
import { ActivationFunction } from '../activation';
import { Perceptron, initRandom, updateOutput } from '../perceptron/perceptron';

// Single Layer Perceptron has 1 hidden layer with one or more perceptrons
// Input layer has one or more inputs
// Output layer has 1 perceptron/output
export type SingleLayerPerceptron = {
    nInputs: number,            //number of inputs
    inputs?: number[],          //input values (input layer)
    hiddenLayer: Perceptron[],  //hidden layer
    outputLayer: Perceptron,    //output layer
    output?: number,            //output
}

// Initialize a perceptron with given set of perceptrons
// Inputs and output are undefined at this point
export const initSLP = (hiddenLayer: Perceptron[], outputLayer: Perceptron): SingleLayerPerceptron =>  {
    const hiddenSize = hiddenLayer.length;
    if (hiddenSize === 0) {
        throw new Error("Hidden layer must have at least one perceptron");
    }
    const nInputs = hiddenLayer[0].weights.length;
    hiddenLayer.forEach((perceptron) => {
        if (perceptron.weights.length !== nInputs) {
            throw new Error("All perceptrons in the hidden layer must have the same number of inputs");
        }
    });
    if (outputLayer.weights.length !== hiddenSize) {
        throw new Error("Output layer perceptron must have the same number of inputs as the number of perceptrons in the hidden layer");
    }
    const slp = {
        hiddenLayer: hiddenLayer,
        outputLayer: outputLayer,
        nInputs: nInputs,
    }
    return slp;
}

// Initialize SLP with random perceptrons (weights and bias) and activation function
// Inputs and output are undefined at this point
// Parameters:
// inputSize -> number of inputs
// hiddenSize -> number of perceptrons in the hidden layer
// func -> activation function for every perceptron
export const initRandomSLP = (inputSize: number, hiddenSize: number, func: ActivationFunction): SingleLayerPerceptron =>  {
    const hiddenLayer: Perceptron[] = [];
    for (let i = 0; i < hiddenSize; i++) {
        hiddenLayer.push(initRandom(inputSize, func));
    }
    const outputLayer = initRandom(hiddenSize, func);
    return initSLP(hiddenLayer, outputLayer);
}

// Forward propagation (update output) for SLP
export const forwardPropagationSLP = (slp: SingleLayerPerceptron, inputs: number[]): SingleLayerPerceptron => {
    if (inputs.length !== slp.nInputs) {
        throw new Error("Input length must match hidden layer weights length");
    }
    slp.inputs = inputs;
    const hiddenLayerOutputs: number[] = [];
    for (let i = 0; i < slp.hiddenLayer.length; i++) {
        let perceptron = slp.hiddenLayer[i];
        perceptron = updateOutput(perceptron, inputs);
        if (perceptron.output === undefined) {
            throw new Error(`Output for hidden layer perceptron ${i} is undefined`);
        }
        hiddenLayerOutputs.push(perceptron.output);
        slp.hiddenLayer[i] = perceptron;
    }
    let perceptron = slp.outputLayer;
    perceptron = updateOutput(perceptron, hiddenLayerOutputs);
    if (perceptron.output === undefined) {
        throw new Error(`Output for output layer perceptron is undefined`);
    }
    slp.outputLayer = perceptron;
    slp.output = perceptron.output;

    return slp;
}