import { describe, it, expect } from 'vitest';
import { perceptronActivation, sigmoid } from './activation';

describe('Activation Function Module', () => {
    it('should return 1 for positive input and 0 for non-positive input', () => {
        expect(perceptronActivation(1)).toBe(1);
        expect(perceptronActivation(0)).toBe(0);
        expect(perceptronActivation(-1)).toBe(0);
        expect(perceptronActivation(0.5)).toBe(1);
        expect(perceptronActivation(-0.5)).toBe(0);
    });

    it('should return a value between 0 and 1 for any input', () => {
        expect(sigmoid(0)).toBeCloseTo(0.5, 5);
        expect(sigmoid(1)).toBeCloseTo(0.73106, 5);
        expect(sigmoid(-1)).toBeCloseTo(0.26894, 5);
        expect(sigmoid(100)).toBeCloseTo(1, 5);
        expect(sigmoid(-100)).toBeCloseTo(0, 5);
    });
});