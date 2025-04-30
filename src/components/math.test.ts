import { describe, it, expect } from 'vitest';
import { dotProduct } from './math';

describe('Math Module', () => {

    it('should calculate the dot product of two arrays', () => {
        const result = dotProduct([1, 2, 3], [4, 5, 6]);
        expect(result).toBe(32); // 1*4 + 2*5 + 3*6 = 32
    });

    it('should throw an error if dot product arrays have different lengths', () => {
        expect(() => dotProduct([1, 2], [1, 2, 3])).toThrow('Arrays must be of the same length');
    });
});