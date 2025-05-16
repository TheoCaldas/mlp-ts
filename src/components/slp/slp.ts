import { ActivationFunction, sigmoidDerivative } from '../activation';
import { Perceptron, initRandom, updateOutput } from '../perceptron/perceptron';
import { squareLoss, squareLossDerivative } from '../loss';

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

// SLP backpropagation step result
export type SLPGradients = {
    outputLayerBias: number,        //output layer bias gradient
    outputLayerWeights: number[],   //output layer weights gradient
    hiddenLayerBias: number[],      //hidden layer bias gradient
    hiddenLayerWeights: number[][], //hidden layer weights gradient      
}

// Initialize a model with given set of perceptrons
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

// Initialize SLP gradients with zeros
export const initSLPGrads = (inputSize: number, hiddenSize: number): SLPGradients =>  {
    return {
        outputLayerBias: 0,
        outputLayerWeights: Array.from({ length: hiddenSize }, () => 0),
        hiddenLayerBias: Array.from({ length: hiddenSize }, () => 0),
        hiddenLayerWeights: Array.from({ length: hiddenSize }, () => Array.from({ length: inputSize }, () => 0)),
    };
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
        // console.log(`Hidden Layer Output (${i}): `, perceptron.output);
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
    // console.log(`Output Layer Output: `, slp.output);
    return slp;
}

// One setp backpropagation for SLP
// Returns the updated SLP and gradients
// Does not update the weights and bias
export const backpropagationSLP = (
    slp: SingleLayerPerceptron, 
    expectedOutput: number, 
    verbose: boolean = false): 
    {slp: SingleLayerPerceptron, grads: SLPGradients} => {

    if (slp.output === undefined) {
        throw new Error("Output is undefined");
    }
    if (slp.inputs === undefined) {
        throw new Error(`Inputs are undefined`);
    }

    // Initialize local gradients
    const grads = initSLPGrads(slp.nInputs, slp.hiddenLayer.length);

    // Calculate the output gradient
    const outputGrad = squareLossDerivative(slp.output, expectedOutput);
    const sigmoidGrad = outputGrad * sigmoidDerivative(slp.output);

    // Calculate the bias gradient for the output layer
    const biasGrad = sigmoidGrad;
    if (verbose) console.log("Output Layer Bias Grad: ", biasGrad);
    grads.outputLayerBias = biasGrad;

    // Calculate the weights gradients for the output layer
    for (let i = 0; i < slp.outputLayer.weights.length; i++) {
        if (slp.hiddenLayer[i].output === undefined) {
            throw new Error(`Weight for output layer perceptron ${i} is undefined`);
        }
        const weightGrad = sigmoidGrad * slp.hiddenLayer[i].output!;
        if (verbose) console.log(`Output Layer Weight Grads (${i}): `, weightGrad);
        grads.outputLayerWeights[i] = weightGrad;
    }
    
    for (let i = 0; i < slp.hiddenLayer.length; i++) {
        // Calculate the bias gradient for the hidden layer
        const hiddenBiasGrad = sigmoidGrad * sigmoidDerivative(slp.hiddenLayer[i].output!) * slp.outputLayer.weights[i]; 
        if (verbose) console.log(`Hidden Layer Bias Grad (${i}): `, hiddenBiasGrad);
        grads.hiddenLayerBias[i] = hiddenBiasGrad;
        
        // Calculate the weights gradients for the hidden layer
        for (let j = 0; j < slp.inputs.length; j++) {
            const hiddenWeightGrad = hiddenBiasGrad * slp.inputs[j];
            if (verbose) console.log(`Hidden Layer Weight Grads (${i}, ${j}): `, hiddenWeightGrad);
            grads.hiddenLayerWeights[i][j] = hiddenWeightGrad;
        }
    }
    
    return {slp, grads};
}