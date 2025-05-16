import { describe, it, expect } from 'vitest';
import { forwardPropagationMLP, initMLP, initMLPGrads, initRandomMLP, MultiLayerPerceptron } from './mlp';
import { initRandom } from '../perceptron/perceptron';
import { sigmoid } from '../activation';

describe('MLP Module', () => {
    describe('initMLP', () => {
        it('should initialize a valid MLP with given hidden and output layers', () => {
            const hiddenLayers = [
                [initRandom(3, sigmoid), initRandom(3, sigmoid)],
                [initRandom(2, sigmoid)],
            ];
            const outputLayer = [initRandom(1, sigmoid)];
            const mlp = initMLP(hiddenLayers, outputLayer);

            expect(mlp.nInputs).toBe(3);
            expect(mlp.nOutputs).toBe(1);
            expect(mlp.nHidden).toBe(2);
            expect(mlp.hiddenLayers).toHaveLength(2);
            expect(mlp.outputLayer).toHaveLength(1);
        });

        it('should throw an error if no hidden layers are provided', () => {
            const outputLayer = [initRandom(1, sigmoid)];
            expect(() => initMLP([], outputLayer)).toThrowError(
                'MLP must have at least one hidden layer'
            );
        });

        it('should throw an error if a hidden layer has no perceptrons', () => {
            const hiddenLayers = [[]];
            const outputLayer = [initRandom(1, sigmoid)];
            expect(() => initMLP(hiddenLayers, outputLayer)).toThrowError(
                'Hidden layer must have at least one perceptron'
            );
        });

        it('should throw an error if layer inputs do not match previous layer outputs', () => {
            const hiddenLayers = [
                [initRandom(3, sigmoid)],
                [initRandom(4, sigmoid)],
            ];
            const outputLayer = [initRandom(1, sigmoid)];
            expect(() => initMLP(hiddenLayers, outputLayer)).toThrowError(
                'Layer inputs must match previous layer outputs'
            );
        });

        it('should throw an error if output layer inputs do not match last hidden layer outputs', () => {
            const hiddenLayers = [
                [initRandom(3, sigmoid)],
                [initRandom(1, sigmoid)],
            ];
            const outputLayer = [initRandom(3, sigmoid)];
            expect(() => initMLP(hiddenLayers, outputLayer)).toThrowError(
                'Output layer perceptron must have the same number of inputs as the number of perceptrons in the last hidden layer'
            );
        });
    });

    describe('initRandomMLP', () => {
        it('should initialize a valid random MLP with given parameters', () => {
            const nInputs = 3;
            const nOutputs = 1;
            const hiddenDimensions = [4, 2];
            const mlp = initRandomMLP(nInputs, nOutputs, hiddenDimensions, sigmoid, sigmoid);

            expect(mlp.nInputs).toBe(nInputs);
            expect(mlp.nOutputs).toBe(nOutputs);
            expect(mlp.nHidden).toBe(hiddenDimensions.length);
            expect(mlp.hiddenLayers).toHaveLength(hiddenDimensions.length);
            expect(mlp.hiddenLayers[0]).toHaveLength(hiddenDimensions[0]);
            expect(mlp.hiddenLayers[1]).toHaveLength(hiddenDimensions[1]);
            expect(mlp.outputLayer).toHaveLength(nOutputs);
        });

        it('should throw an error if hidden dimensions array is empty', () => {
            const nInputs = 3;
            const nOutputs = 1;
            expect(() =>
                initRandomMLP(nInputs, nOutputs, [], sigmoid, sigmoid)
            ).toThrowError('MLP must have at least one hidden layer');
        });
    });

    describe('initMLPGrads', () => {
        it('should initialize gradients with zeros for a valid MLP', () => {
            const hiddenLayers = [
                [initRandom(3, sigmoid), initRandom(3, sigmoid)],
                [initRandom(2, sigmoid)],
            ];
            const outputLayer = [initRandom(1, sigmoid)];
            const mlp = initMLP(hiddenLayers, outputLayer);
            const grads = initMLPGrads(mlp);

            expect(grads.outputLayerBias).toHaveLength(mlp.nOutputs);
            expect(grads.outputLayerBias.every((bias) => bias === 0)).toBe(true);

            expect(grads.outputLayerWeights).toHaveLength(mlp.nOutputs);
            grads.outputLayerWeights.forEach((weights, i) => {
                expect(weights).toHaveLength(mlp.outputLayer[i].weights.length);
                expect(weights.every((weight) => weight === 0)).toBe(true);
            });

            expect(grads.hiddenLayersBias).toHaveLength(mlp.nHidden);
            grads.hiddenLayersBias.forEach((layerBias, i) => {
                expect(layerBias).toHaveLength(mlp.hiddenLayers[i].length);
                expect(layerBias.every((bias) => bias === 0)).toBe(true);
            });

            expect(grads.hiddenLayersWeights).toHaveLength(mlp.nHidden);
            grads.hiddenLayersWeights.forEach((layerWeights, i) => {
                expect(layerWeights).toHaveLength(mlp.hiddenLayers[i].length);
                layerWeights.forEach((weights, j) => {
                    expect(weights).toHaveLength(mlp.hiddenLayers[i][j].weights.length);
                    expect(weights.every((weight) => weight === 0)).toBe(true);
                });
            });
        });
    });

    describe('forwardPropagationMLP', () => {
        it('should correctly propagate inputs through the MLP', () => {
            const hiddenLayers = [
                [initRandom(3, sigmoid), initRandom(3, sigmoid)],
                [initRandom(2, sigmoid)],
            ];
            const outputLayer = [initRandom(1, sigmoid)];
            const mlp = initMLP(hiddenLayers, outputLayer);

            const inputs = [0.5, 0.8, 0.2];
            const updatedMLP = forwardPropagationMLP(mlp, inputs);

            expect(updatedMLP.inputs).toEqual(inputs);
            expect(updatedMLP.hiddenLayers[0][0].output).not.toBeUndefined();
            expect(updatedMLP.hiddenLayers[0][1].output).not.toBeUndefined();
            expect(updatedMLP.hiddenLayers[1][0].output).not.toBeUndefined();
            expect(updatedMLP.outputs).toHaveLength(1);
            expect(updatedMLP.outputs![0]).not.toBeUndefined();
        });

        it('should throw an error if input length does not match MLP input size', () => {
            const hiddenLayers = [
                [initRandom(3, sigmoid), initRandom(3, sigmoid)],
                [initRandom(2, sigmoid)],
            ];
            const outputLayer = [initRandom(1, sigmoid)];
            const mlp = initMLP(hiddenLayers, outputLayer);

            const invalidInputs = [0.5, 0.8]; // Length does not match nInputs
            expect(() => forwardPropagationMLP(mlp, invalidInputs)).toThrowError(
                'Input length must match first hidden layer weights length'
            );
        });

        it('should update all perceptrons in hidden and output layers with valid outputs', () => {
            const hiddenLayers = [
                [initRandom(3, sigmoid), initRandom(3, sigmoid)],
                [initRandom(2, sigmoid)],
            ];
            const outputLayer = [initRandom(1, sigmoid)];
            const mlp = initMLP(hiddenLayers, outputLayer);

            const inputs = [0.1, 0.4, 0.7];
            const updatedMLP = forwardPropagationMLP(mlp, inputs);

            updatedMLP.hiddenLayers.forEach((layer) => {
                layer.forEach((perceptron) => {
                    expect(perceptron.output).not.toBeUndefined();
                });
            });

            updatedMLP.outputLayer.forEach((perceptron) => {
                expect(perceptron.output).not.toBeUndefined();
            });
        });
    });
});