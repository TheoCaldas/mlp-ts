import { describe, it, expect } from 'vitest';
import { init, initRandom, updateOutput, dotProduct } from './perceptron';

describe('Perceptron Module', () => {
    it('should initialize a perceptron with given weights and bias', () => {
        const activationFunc = (x: number) => (x > 0 ? 1 : 0);
        const perceptron = init([0.5, -0.5], 0.1, activationFunc);

        expect(perceptron.weights).toEqual([0.5, -0.5]);
        expect(perceptron.bias).toBe(0.1);
        expect(perceptron.activationFunc).toBe(activationFunc);
    });

    it('should initialize a perceptron with random weights and bias', () => {
        const activationFunc = (x: number) => (x > 0 ? 1 : 0);
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

    it('should calculate the dot product of two arrays', () => {
        const result = dotProduct([1, 2, 3], [4, 5, 6]);
        expect(result).toBe(32); // 1*4 + 2*5 + 3*6 = 32
    });

    it('should throw an error if dot product arrays have different lengths', () => {
        expect(() => dotProduct([1, 2], [1, 2, 3])).toThrow('Arrays must be of the same length');
    });

    it('should update the perceptron output correctly', () => {
        const activationFunc = (x: number) => (x > 0 ? 1 : 0);
        const perceptron = init([0.5, -0.5], 0.1, activationFunc);

        const updatedPerceptron = updateOutput(perceptron, [1, 1]);
        expect(updatedPerceptron.output).toBe(1); // z = (1*0.5 + 1*(-0.5)) + 0.1 = 0.1 > 0
    });

    it('should throw an error if input length does not match weights length', () => {
        const activationFunc = (x: number) => (x > 0 ? 1 : 0);
        const perceptron = init([0.5, -0.5], 0.1, activationFunc);

        expect(() => updateOutput(perceptron, [1])).toThrow('Input length must match weights length');
    });
});