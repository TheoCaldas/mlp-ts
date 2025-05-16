import { ActivationFunction, sigmoidDerivative } from '../activation';
import { Perceptron, initRandom, updateOutput } from '../perceptron/perceptron';
import { squareLoss, squareLossDerivative } from '../loss';
import { initZeroArray } from '../math';

// MultiLayer Perceptron
// One or more hidden layers with one or more perceptrons
// Input layer with one or more inputs
// Output layer with one or more inputs
export type MultiLayerPerceptron = {
    nInputs: number,                //number of inputs
    inputs?: number[],              //input values (input layer)

    nOutputs: number,               //number of outputs
    outputs?: number[],             //output values (output layer outputs)
    outputLayer: Perceptron[],      //output layer, index as [perceptron]

    nHidden: number,                //number of hidden layers
    hiddenLayers: Perceptron[][],   //hidden layers, index as [layer, perceptron]
}

// MLP backpropagation step result
export type MLPGradients = {
    outputLayerBias: number[],         //output layer bias gradient array, index as [perceptron]
    outputLayerWeights: number[][],    //output layer weights gradient 2d-matrix, index as [perceptron, input]
    hiddenLayersBias: number[][],      //hidden layer bias gradient 2d-matrix, index as [layer, perceptron]
    hiddenLayersWeights: number[][][], //hidden layer weights gradient 3d-matrix, index as [layer, perceptron, input]
}

// Initialize MLP with parameters
// Inputs and outputs are undefined at this point
export const initMLP = (hiddenLayers: Perceptron[][], outputLayer: Perceptron[]): MultiLayerPerceptron =>  {
    if (hiddenLayers.length === 0) {
        throw new Error("MLP must have at least one hidden layer");
    }
    hiddenLayers.forEach((hiddenLayer, i) => {
        if (hiddenLayer.length === 0) {
            throw new Error("Hidden layer must have at least one perceptron");
        }
        if (i > 0) {
            hiddenLayer.forEach((perceptron) => {
                if (perceptron.weights.length !== hiddenLayers[i-1].length) {
                    throw new Error("Layer inputs must match previous layer outputs");
                }
            });
        }
    });
    outputLayer.forEach((perceptron) => {
        if (perceptron.weights.length !== hiddenLayers.at(-1)!.length) {
            throw new Error("Output layer perceptron must have the same number of inputs as the number of perceptrons in the last hidden layer");
        }
    });

    const nInputs = hiddenLayers[0][0].weights.length;
    const nOutputs = outputLayer.length;
    const nHidden = hiddenLayers.length;
    return {
        hiddenLayers: hiddenLayers,
        outputLayer: outputLayer,
        nInputs: nInputs,
        nOutputs: nOutputs,
        nHidden: nHidden,
    };
}

// Initialize MLP with random perceptrons (weights and bias) and activation functions
// Inputs and outputs are undefined at this point
// Parameters:

export const initRandomMLP = (
    nInputs: number,                //number of inputs
    nOutputs: number,               //number of outputs
    hiddenDimensions: number[],     //array of number of perceptrons in each hidden layer
    hiddenFunc: ActivationFunction, //activation function for hidden layers
    outputFunc: ActivationFunction, //activation function for output layer
): MultiLayerPerceptron =>  {

    const hiddenLayers: Perceptron[][] = [];
    for (let i = 0; i < hiddenDimensions.length; i++) {
        const hiddenLayer: Perceptron[] = [];
        for (let j = 0; j < hiddenDimensions[i]; j++) {
            if (i === 0) {
                hiddenLayer.push(initRandom(nInputs, hiddenFunc));
            } else {
                hiddenLayer.push(initRandom(hiddenDimensions[i-1], hiddenFunc));
            }
        }
        hiddenLayers.push(hiddenLayer);
    }
    const outputLayer: Perceptron[] = [];
    for (let i = 0; i < nOutputs; i++) {
        outputLayer.push(initRandom(hiddenDimensions.at(-1)!, outputFunc));
    }
    return initMLP(hiddenLayers, outputLayer);
}

// Initialize MLP gradients with zeros
export const initMLPGrads = (mlp: MultiLayerPerceptron): MLPGradients =>  {
    return {
        outputLayerBias: initZeroArray(mlp.nOutputs),
        outputLayerWeights: mlp.outputLayer.map((perceptron) => initZeroArray(perceptron.weights.length)),
        hiddenLayersBias: mlp.hiddenLayers.map((layer) => initZeroArray(layer.length)),
        hiddenLayersWeights: mlp.hiddenLayers.map((layer) => layer.map((perceptron) => initZeroArray(perceptron.weights.length))),
    };
}

// MLP Forward propagation (update outputs)
export const forwardPropagationMLP = (mlp: MultiLayerPerceptron, inputs: number[]): MultiLayerPerceptron => {
    if (inputs.length !== mlp.nInputs) {
        throw new Error("Input length must match first hidden layer weights length");
    }
    mlp.inputs = inputs;
    const hiddenLayersOutputs: number[][] = [];
    for (let i = 0; i < mlp.hiddenLayers.length; i++) {
        let hiddenLayer = mlp.hiddenLayers[i];
        const layerOutputs: number[] = [];

        for (let j = 0; j < hiddenLayer.length; j++) {
            let perceptron = hiddenLayer[j];
            perceptron = updateOutput(perceptron, (i == 0) ? inputs : hiddenLayersOutputs[i-1]);
            layerOutputs.push(perceptron.output!);
            hiddenLayer[j] = perceptron;
        }
        hiddenLayersOutputs.push(layerOutputs);
        mlp.hiddenLayers[i] = hiddenLayer;
    }
    
    const outputLayerOutputs: number[] = [];
    for (let j = 0; j < mlp.outputLayer.length; j++) {
        let perceptron = mlp.outputLayer[j];
        perceptron = updateOutput(perceptron, hiddenLayersOutputs.at(-1)!);
        outputLayerOutputs.push(perceptron.output!);
        mlp.outputLayer[j] = perceptron;
    }

    mlp.outputs = outputLayerOutputs;
    return mlp;
}

// // One setp backpropagation for MLP
// // Returns the updated MLP and gradients
// // Does not update the weights and bias
// export const backpropagationMLP = (
//     mlp: MultiLayerPerceptron, 
//     expectedOutputs: number[], 
//     verbose: boolean = false): 
//     {mlp: MultiLayerPerceptron, grads: MLPGradients} => {

//     if (mlp.outputs === undefined) {
//         throw new Error("Output is undefined");
//     }
//     if (mlp.inputs === undefined) {
//         throw new Error(`Inputs are undefined`);
//     }

//     // Initialize local gradients
//     const grads = initMLPGrads(mlp);

//     // Calculate the output gradient
//     const outputGrads = mlp.outputs.map((output, i) => squareLossDerivative(output, expectedOutputs[i]));
//     const outputSigmoidGrads = mlp.outputs.map((output, i) => outputGrads[i] * sigmoidDerivative(output)); 

//     // Calculate the bias gradient for the output layer
//     const outputLayerBiasGrads = outputSigmoidGrads;
//     if (verbose) console.log("Output Layer Bias Grads: ", outputLayerBiasGrads);
//     grads.outputLayerBias = outputLayerBiasGrads;

//     // Calculate the weights gradients for the output layer
//     for (let i = 0; i < mlp.outputLayer.length; i++) {
//         for (let j = 0; j < mlp.outputLayer[i].weights.length; j++) {
//             const outputLayerWeightGrads = outputSigmoidGrads[i] * mlp.hiddenLayers.at(-1)![j].output!;
//             if (verbose) console.log(`Output Layer Weight Grads (${i}, ${j}): `, outputLayerWeightGrads);
//             grads.outputLayerWeights[i][j] = outputLayerWeightGrads;
//         }
//     }

//     // Calculate the bias gradient for the hidden layers
//     // Calculate the weights gradient for the hidden layers

    
//     return {mlp: mlp, grads};
// }