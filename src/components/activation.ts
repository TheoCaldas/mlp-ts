export type ActivationFunction = (x: number) => number;

// This is a step function that returns 1 if x > 0, otherwise returns 0
export const perceptronActivation: ActivationFunction = (x: number) => {
    return x > 0 ? 1 : 0;
};