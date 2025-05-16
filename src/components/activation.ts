export type ActivationFunction = (x: number) => number;

// This is a step function that returns 1 if x > 0, otherwise returns 0
export const perceptronActivation: ActivationFunction = (x: number) => {
    return x > 0 ? 1 : 0;
};

// Sigmoid activation function
export const sigmoid: ActivationFunction = (x: number) => {
    return 1 / (1 + Math.exp(-x));
};

// Sigmoid derivative function
export const sigmoidDerivative: ActivationFunction = (activationOutput: number) => {
    return activationOutput * (1 - activationOutput);
};

// ReLU (Rectified Linear Unit) activation function
export const reLU: ActivationFunction = (x: number) => {
    return Math.max(0, x);
};

// ReLU derivative function
export const reLUDerivative: ActivationFunction = (activationOutput: number) => {
    return activationOutput > 0 ? 1 : 0;
}