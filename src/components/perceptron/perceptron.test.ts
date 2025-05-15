import { describe, it, expect } from 'vitest';
import { init, initRandom, initXavier, updateOutput } from './perceptron';
import { perceptronActivation } from '../activation';

const activationFunc = perceptronActivation;

describe('Perceptron Module', () => {

    it('should initialize a perceptron with given weights and bias', () => {    
        const perceptron = init([0.5, -0.5], 0.1, activationFunc);

        expect(perceptron.weights).toEqual([0.5, -0.5]);
        expect(perceptron.bias).toBe(0.1);
        expect(perceptron.activationFunc).toBe(activationFunc);
    });

    it('should initialize a perceptron with random weights and bias', () => {
        const perceptron = initRandom(3, activationFunc);

        expect(perceptron.weights).toHaveLength(3);
        perceptron.weights.forEach(weight => {
            expect(weight).toBeGreaterThanOrEqual(0);
            expect(weight).toBeLessThan(1);
        });
        expect(perceptron.bias).toBeGreaterThanOrEqual(0);
        expect(perceptron.bias).toBeLessThan(1);
        expect(perceptron.activationFunc).toBe(activationFunc);
    });

    it('should initialize a perceptron with xavier distribution and zero biases', () => {
        const nInputs = 3;
        const nOutputs = 2;
        const perceptron = initXavier(nInputs, nOutputs, activationFunc);

        expect(perceptron.weights).toHaveLength(nInputs);
        const limit = Math.sqrt(6 / (nInputs + nOutputs));
        perceptron.weights.forEach(weight => {
            expect(weight).toBeGreaterThanOrEqual(-limit);
            expect(weight).toBeLessThanOrEqual(limit);
        });
        expect(perceptron.bias).toBe(0);
        expect(perceptron.activationFunc).toBe(activationFunc);
    });

    it('should update the perceptron output correctly', () => {
        const perceptron = init([0.5, -0.5], 0.1, activationFunc);

        const updatedPerceptron = updateOutput(perceptron, [1, 1]);
        expect(updatedPerceptron.output).toBe(1); // z = (1*0.5 + 1*(-0.5)) + 0.1 = 0.1 > 0
    });

    it('should throw an error if input length does not match weights length', () => {
        const perceptron = init([0.5, -0.5], 0.1, activationFunc);

        expect(() => updateOutput(perceptron, [1])).toThrow('Input length must match weights length');
    });
});