import { sigmoid } from "../activation";
import { backpropagationSLP, forwardPropagationSLP, initRandomSLP, initSLP, SingleLayerPerceptron } from "./slp";
import { squareLoss, squareLossDerivative } from "../loss";
import fs from "fs";
import { init } from "../perceptron/perceptron";

// Supervised training of SLP
// Data is an array of items which is an array of inputs, labels are the expected outputs
// Learning rate is the step size for weight updates, between 0 and 1
export type SupervisedSLP = {
    data: number[][],
    labels: number[],
    slp: SingleLayerPerceptron,
    learningRate: number,
    itemCount: number,
}

// Initialize a supervised SLP with given data, labels and learning rate
// Specify the size of the hidden layer
// The output layer has 1 perceptron/output
export const initTrainingSLP = (
    data: number[][], 
    labels: number[], 
    hiddenLayerSize: number,
    learningRate: number): SupervisedSLP => {

    const itemCount = data.length;
    if (itemCount === 0) {
        throw new Error("Data cannot be empty");
    }
    if (itemCount !== labels.length) {
        throw new Error("Data and labels must have the same length");
    }
    if (learningRate <= 0 || learningRate > 1) {
        throw new Error("Learning rate must be between 0 and 1");
    }
    const nInputs = data[0].length;
    const slp = initRandomSLP(nInputs, hiddenLayerSize, sigmoid);
    return {
        data: data,
        labels: labels,
        slp: slp,
        learningRate: learningRate,
        itemCount: itemCount,
    };
}

// Train SLP using the given data and labels
// Update weights and bias at each step
// Parameters:
// model -> SLP model
// hiddenFactor -> constant factor to avoid vanishing hidden gradients (default is 1)
// verbose -> boolean to enable verbose logging (default is false)
export const trainSLP = (
    model: SupervisedSLP, 
    hiddenFactor: number = 1, 
    verbose: boolean = false): SupervisedSLP => {
    for (let i = 0; i < model.itemCount; i++) {
        const inputs = model.data[i];
        const expectedOutput = model.labels[i];
        
        // Forward propagation
        model.slp = forwardPropagationSLP(model.slp, inputs);
        if (model.slp.output === undefined) {
            throw new Error(`Output for item ${i} is undefined`);
        }
        const output = model.slp.output;
        const error = squareLoss(output, expectedOutput);
        const errorDerivative = squareLossDerivative(output, expectedOutput);


        // Backpropagation
        const {slp, grads} = backpropagationSLP(model.slp, expectedOutput, verbose);
        model.slp = slp;

        // Optional: Log the training process
        if (verbose){
            console.log(`Training item ${i + 1}/${model.itemCount}`);
            console.log(`Inputs: ${inputs}`);
            console.log(`Expected Output: ${expectedOutput}`);
            console.log(`Actual Output: ${output}`);
            console.log(`Error: ${error}`);
            console.log(`Error Derivative: ${errorDerivative}`);
        }

        // Update weights and bias
        const n = model.learningRate * -1;
        model.slp.outputLayer.bias += n * grads.outputLayerBias;

        if (verbose) console.log(`Output Layer Bias: ${model.slp.outputLayer.bias}`);
        for (let i = 0; i < model.slp.outputLayer.weights.length; i++) {
            model.slp.outputLayer.weights[i] += n * grads.outputLayerWeights[i];
            if (verbose) console.log(`Output Layer Weights ${i}: ${model.slp.outputLayer.weights[i]}`);

            model.slp.hiddenLayer[i].bias += n * grads.hiddenLayerBias[i] * hiddenFactor;
            if (verbose) console.log(`Hidden Layer Bias ${i}: ${model.slp.hiddenLayer[i].bias}`);

            for (let j = 0; j < model.slp.hiddenLayer[i].weights.length; j++) {
                model.slp.hiddenLayer[i].weights[j] += n * grads.hiddenLayerWeights[i][j] * hiddenFactor;
                if (verbose) console.log(`Hidden Layer Weights (${i},${j}): ${model.slp.hiddenLayer[i].weights[j]}`);
            }
        }
        if (verbose) console.log("-----------------------------");
    }
    return model;
}

// Predict the output for a given item using the trained model
// Use threshold of 0.5 to classify the output (map to 0 or 1)
export const predictSLP = (model: SupervisedSLP, item: number[]): number => {
    const slp = model.slp;
    const updatedSLP = forwardPropagationSLP(slp, item);
    if (updatedSLP.output === undefined) {
        throw new Error("Output is undefined");
    }
    return updatedSLP.output < 0.5 ? 0 : 1;
}

// Test the model with given data and labels
export const testSLP = (model: SupervisedSLP, data: number[][], labels: number[], verbose: boolean = false): number => {
    if (data.length !== labels.length) {
        throw new Error("Data and labels must have the same length");
    }
    if (verbose) console.log("Testing the model...\n");

    let score = 0;
    for (let i = 0; i < data.length; i++) {
        const prediction = predictSLP(model, data[i]);
        
        if (prediction !== labels[i]) {
            if (verbose){
                console.log("Wrong Prediction!\n");
                console.log(`Inputs: ${data[i]}`);
                console.log(`Expected Output: ${labels[i]}`);
                console.log(`Predicted Output: ${prediction}\n`);
            }
        } else {
            score++;
        }

        if (verbose){
            console.log(`Current Score: ${score}/${i + 1}`);
            console.log("-----------------------------");
        }
    }

    const accuracy = score / data.length;
    if (verbose) console.log(`\nFinal Accuracy: ${accuracy}`);
    
    return accuracy;
}

export const exportModelSLP = (model: SupervisedSLP, filePath: string) => {
    const modelData = {
        outputLayerBias: model.slp.outputLayer.bias,
        outputLayerWeights: model.slp.outputLayer.weights,
        hiddenLayerBias: model.slp.hiddenLayer.map(perceptron => perceptron.bias),
        hiddenLayerWeights: model.slp.hiddenLayer.map(perceptron => perceptron.weights),
    };
    const jsonString = JSON.stringify(modelData, null, 2);
    fs.writeFileSync(filePath, jsonString);
    console.log(`Model exported to ${filePath}`);
}

export const importModelSLP = (filePath: string): SupervisedSLP => {
    const jsonString = fs.readFileSync(filePath, 'utf-8');
    const modelData = JSON.parse(jsonString);

    const outputLayer = init(
        modelData.outputLayerWeights, 
        modelData.outputLayerBias, 
        sigmoid
    );

    const hiddenLayer = modelData.hiddenLayerWeights.map((weights: number[], i: number) => {
        return init(weights, modelData.hiddenLayerBias[i], sigmoid);
    });

    const slp = initSLP(hiddenLayer, outputLayer);
    console.log(`Model imported from ${filePath}`);
    return {
        data: [],
        labels: [],
        slp: slp,
        learningRate: 0,
        itemCount: 0,
    };
}