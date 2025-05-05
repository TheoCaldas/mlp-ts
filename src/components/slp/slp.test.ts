import { initSLP, initRandomSLP, forwardPropagationSLP, backPropagationSLP, initSLPGrads } from './slp';
import { initRandom, init } from '../perceptron/perceptron';
import { sigmoid } from '../activation';
import { describe, it, expect } from 'vitest';

describe('Single Layer Perceptron Module', () => {

    it('should initialize SLP with valid hidden and output layers', () => {
        const hiddenLayer = [
            initRandom(3, sigmoid),
            initRandom(3, sigmoid),
        ];
        const outputLayer = initRandom(2, sigmoid);

        const slp = initSLP(hiddenLayer, outputLayer);

        expect(slp.hiddenLayer).toHaveLength(2);
        expect(slp.outputLayer.weights).toHaveLength(2);
        expect(slp.nInputs).toBe(3);
    });

    it('should throw error if hidden layer perceptrons have inconsistent input sizes', () => {
        const hiddenLayer = [
            initRandom(3, sigmoid),
            initRandom(4, sigmoid),
        ];
        const outputLayer = initRandom(2, sigmoid);

        expect(() => initSLP(hiddenLayer, outputLayer)).toThrowError(
            'All perceptrons in the hidden layer must have the same number of inputs'
        );
    });

    it('should throw error if output layer perceptron input size does not match hidden layer size', () => {
        const hiddenLayer = [
            initRandom(3, sigmoid),
            initRandom(3, sigmoid),
        ];
        const outputLayer = initRandom(3, sigmoid);

        expect(() => initSLP(hiddenLayer, outputLayer)).toThrowError(
            'Output layer perceptron must have the same number of inputs as the number of perceptrons in the hidden layer'
        );
    });

    it('should initialize random SLP with correct structure', () => {
        const inputSize = 3;
        const hiddenSize = 2;

        const slp = initRandomSLP(inputSize, hiddenSize, sigmoid);

        expect(slp.hiddenLayer).toHaveLength(hiddenSize);
        expect(slp.hiddenLayer[0].weights).toHaveLength(inputSize);
        expect(slp.outputLayer.weights).toHaveLength(hiddenSize);
        expect(slp.nInputs).toBe(inputSize);
    });

    it('should perform forward propagation and update outputs', () => {
        const inputSize = 3;
        const hiddenSize = 2;
        const inputs = [1, 0, -1];

        const slp = initRandomSLP(inputSize, hiddenSize, sigmoid);

        const updatedSLP = forwardPropagationSLP(slp, inputs);

        expect(updatedSLP.inputs).toEqual(inputs);
        updatedSLP.hiddenLayer.forEach((perceptron) => {
            expect(perceptron.output).not.toBeUndefined();
        });
        expect(updatedSLP.outputLayer.output).not.toBeUndefined();
        expect(updatedSLP.output).toBeDefined();
    });

    it('should throw error if input length does not match SLP input size during forward propagation', () => {
        const inputSize = 3;
        const hiddenSize = 2;
        const inputs = [1, 0]; // Incorrect input size

        const slp = initRandomSLP(inputSize, hiddenSize, sigmoid);

        expect(() => forwardPropagationSLP(slp, inputs)).toThrowError(
            'Input length must match hidden layer weights length'
        );
    });

    it('should do a correct forward propagation', () => {
        const inputs = [0.5, 1];
        const hiddenLayer = [
            init([1, 1], 0, sigmoid),
            init([0, 1], 0.5, sigmoid),
            init([0.5, 0.5], 1, sigmoid),
        ];
        const outputLayer = init([0, 0.5, 1], 0, sigmoid);

        const slp = initSLP(hiddenLayer, outputLayer);
        const updatedSLP = forwardPropagationSLP(slp, inputs);
        expect(updatedSLP.hiddenLayer[0].output).toBeCloseTo(0.81, 1);
        expect(updatedSLP.hiddenLayer[1].output).toBeCloseTo(0.81, 1);
        expect(updatedSLP.hiddenLayer[2].output).toBeCloseTo(0.85, 1);
        expect(updatedSLP.outputLayer.output).toBeCloseTo(0.77, 1);
    });

    it('should initialize SLP gradients with zeros', () => {
        const inputSize = 3;
        const hiddenSize = 2;

        const grads = initSLPGrads(inputSize, hiddenSize);

        expect(grads.outputLayerBias).toBe(0);
        expect(grads.outputLayerWeights).toHaveLength(hiddenSize);
        grads.outputLayerWeights.forEach((weight) => {
            expect(weight).toBe(0);
        });

        expect(grads.hiddenLayerBias).toHaveLength(hiddenSize);
        grads.hiddenLayerBias.forEach((bias) => {
            expect(bias).toBe(0);
        });

        expect(grads.hiddenLayerWeights).toHaveLength(hiddenSize);
        grads.hiddenLayerWeights.forEach((weights) => {
            expect(weights).toHaveLength(inputSize);
            weights.forEach((weight) => {
                expect(weight).toBe(0);
            });
        });
    });

    it('should do a correct backpropagation', () => {
        const inputs = [0.5, 1];
        const hiddenLayer = [
            init([1, 1], 0, sigmoid),
            init([0, 1], 0.5, sigmoid),
            init([0.5, 0.5], 1, sigmoid),
        ];
        const outputLayer = init([0, 0.5, 1], 0, sigmoid);
        const expectedOutput = 1;
        const expectedGrads = {
            outputLayerBias: 0.23,
            outputLayerWeights: [0.18, 0.18, 0.20],
            hiddenLayerBias: [0.25, 0.24, 0.25],
            hiddenLayerWeights: [
                [0.12, 0.25],
                [0.12, 0.24],
                [0.12, 0.25],
            ],
        };

        const slp = initSLP(hiddenLayer, outputLayer);
        const updatedSLP = forwardPropagationSLP(slp, inputs);
        const {slp: _, grads} = backPropagationSLP(updatedSLP, expectedOutput);

        expect(grads.outputLayerBias).toBeCloseTo(expectedGrads.outputLayerBias, 1);
        expectedGrads.outputLayerWeights.forEach((weight, index) => {
            expect(grads.outputLayerWeights[index]).toBeCloseTo(weight, 1);
        });
        expectedGrads.hiddenLayerBias.forEach((bias, index) => {
            expect(grads.hiddenLayerBias[index]).toBeCloseTo(bias, 1);
        });
        expectedGrads.hiddenLayerWeights.forEach((weights, index) => {
            weights.forEach((weight, weightIndex) => {
                expect(grads.hiddenLayerWeights[index][weightIndex]).toBeCloseTo(weight, 1);
            });
        }
        );
    });
});