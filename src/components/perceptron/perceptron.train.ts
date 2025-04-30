import { Perceptron, init, initRandom, updateOutput } from "./perceptron";
import { perceptronActivation } from "../activation";
import fs from "fs";

// Supervised training of a single perceptron
// Data is an array of items which is an array of inputs, labels are the expected outputs
// Learning rate is the step size for weight updates, between 0 and 1
export type SupervisedPerceptron = {
    data: number[][],
    labels: number[],
    perceptron: Perceptron,
    learningRate: number,
    itemCount: number,
}

// Initialize a supervised perceptron with given data, labels and learning rate
export const initTraining = (data: number[][], labels: number[], learningRate: number): SupervisedPerceptron => {
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
    const perceptron = initRandom(nInputs, perceptronActivation);
    return {
        data: data,
        labels: labels,
        perceptron: perceptron,
        learningRate: learningRate,
        itemCount: itemCount,
    };
}

// Train the perceptron using the given data and labels
export const train = (model: SupervisedPerceptron, verbose: boolean = false): SupervisedPerceptron => {
    for (let i = 0; i < model.itemCount; i++) {
        const inputs = model.data[i];
        const expectedOutput = model.labels[i];
        let perceptron = model.perceptron;
        perceptron = updateOutput(perceptron, inputs);
        if (perceptron.output === undefined) {
            throw new Error(`Output for item ${i} is undefined`);
        }
        const output = perceptron.output;
        const error = expectedOutput - output;

        // Update weights and bias
        for (let j = 0; j < perceptron.weights.length; j++) {
            perceptron.weights[j] += model.learningRate * error * inputs[j];
        }
        perceptron.bias += model.learningRate * error;

        // Update the perceptron in the model
        model.perceptron = perceptron;

        // Optional: Log the training process
        if (verbose){
            console.log(`Training item ${i + 1}/${model.itemCount}`);
            console.log(`Inputs: ${inputs}`);
            console.log(`Expected Output: ${expectedOutput}`);
            console.log(`Actual Output: ${output}`);
            console.log(`Weights: ${perceptron.weights}`);
            console.log(`Bias: ${perceptron.bias}`);
            console.log(`Error: ${error}`);
            console.log("-----------------------------");
        }
    }
    return model;
}

// Predict the output for a given item using the trained perceptron
export const predict = (model: SupervisedPerceptron, item: number[]): number => {
    const perceptron = model.perceptron;
    const updatedPerceptron = updateOutput(perceptron, item);
    if (updatedPerceptron.output === undefined) {
        throw new Error("Output is undefined");
    }
    return updatedPerceptron.output;
}

// Test the model with given data and labels
export const test = (model: SupervisedPerceptron, data: number[][], labels: number[], verbose: boolean = false): number => {
    if (data.length !== labels.length) {
        throw new Error("Data and labels must have the same length");
    }
    if (verbose) console.log("Testing the model...\n");

    let score = 0;
    for (let i = 0; i < data.length; i++) {
        const prediction = predict(model, data[i]);
        
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

export const exportModel = (model: SupervisedPerceptron, filePath: string) => {
    const modelData = {
        weights: model.perceptron.weights,
        bias: model.perceptron.bias,
    };
    const jsonString = JSON.stringify(modelData, null, 2);
    fs.writeFileSync(filePath, jsonString);
    console.log(`Model exported to ${filePath}`);
}

export const importModel = (filePath: string): SupervisedPerceptron => {
    const jsonString = fs.readFileSync(filePath, 'utf-8');
    const modelData = JSON.parse(jsonString);
    const perceptron = init(modelData.weights, modelData.bias, perceptronActivation);
    console.log(`Model imported from ${filePath}`);
    return {
        data: [],
        labels: [],
        perceptron: perceptron,
        learningRate: 0,
        itemCount: 0,
    };
}