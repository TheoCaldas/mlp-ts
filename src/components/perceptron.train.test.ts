import { initTraining, train, predict } from "./perceptron.train";
import { describe, it, expect } from 'vitest';

describe("Perceptron Training Module", () => {
    it("should initialize a supervised perceptron correctly", () => {
        const data = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ];
        const labels = [0, 0, 0, 1];
        const learningRate = 0.1;

        const model = initTraining(data, labels, learningRate);

        expect(model.data).toEqual(data);
        expect(model.labels).toEqual(labels);
        expect(model.learningRate).toBe(learningRate);
        expect(model.itemCount).toBe(data.length);
        expect(model.perceptron.weights.length).toBe(data[0].length);
    });

    it("should throw an error if data is empty during initialization", () => {
        expect(() => initTraining([], [0], 0.1)).toThrow("Data cannot be empty");
    });

    it("should throw an error if data and labels lengths do not match", () => {
        const data = [[0, 0], [1, 1]];
        const labels = [0];
        expect(() => initTraining(data, labels, 0.1)).toThrow("Data and labels must have the same length");
    });

    it("should throw an error if learning rate is out of bounds", () => {
        const data = [[0, 0]];
        const labels = [0];
        expect(() => initTraining(data, labels, 0)).toThrow("Learning rate must be between 0 and 1");
        expect(() => initTraining(data, labels, 1.1)).toThrow("Learning rate must be between 0 and 1");
    });

    it("should train the perceptron and update weights and bias", () => {
        const data = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ];
        const labels = [0, 0, 0, 1];
        const learningRate = 0.1;

        let model = initTraining(data, labels, learningRate);
        
        const initialWeights = model.perceptron.weights.slice();
        const initialBias = model.perceptron.bias;

        model = train(model);

        expect(model.perceptron.weights).not.toEqual(initialWeights); // Weights should be updated
        expect(model.perceptron.bias).not.toBe(initialBias); // Bias should be updated
    });

    // This test has a low probability of not passing
    it("should predict the correct output after training", () => {
        const data = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],

            [0.1, 0.2],
            [0.1, 0.9],
            [0.9, 0.2],
            [0.9, 0.99],

            [0.01, 0.3],
            [0.3, 0.6],
            [0.8, 0.1],
            [0.6, 0.7],

            [0.4, 0.1],
            [0.3, 0.8],
            [0.7, 0],
            [1, 0.6],

            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],

            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ];
        const labels = [
            0, 0, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1,
            0, 0, 0, 1,
        ]; 
        const learningRate = 0.1;

        let model = initTraining(data, labels, learningRate);
        model = train(model);

        expect(predict(model, [1, 1])).toBe(1);
        expect(predict(model, [0, 0])).toBe(0);
    });
});