export type LossFunction = (output: number, expected: number) => number;

export const squareLoss: LossFunction = (output: number, expected: number) => {
    return Math.pow(expected - output, 2);
};

export const squareLossDerivative = (output: number, expected: number) => {
    return 2 * (output - expected);
};

