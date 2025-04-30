export type ActivationFunction = (x: number) => number;

// This is a step function that returns 1 if x > 0, otherwise returns 0
export const perceptronActivation: ActivationFunction = (x: number) => {
    return x > 0 ? 1 : 0;
};

export const sigmoid: ActivationFunction = (x: number) => {
    return 1 / (1 + Math.exp(-x));
};
// Returns the final decision for sigmoid output
// If the output is greater than the threshold, return 1
// If the output is less than or equal to the threshold, return -1
export const sigmoidDecision = (x: number, t: number): number => {
    return x > t ? 1 : -1;
}
