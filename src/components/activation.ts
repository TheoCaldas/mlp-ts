export type ActivationFunction = (x: number) => number;

// This is a step function that returns 1 if x > 0, otherwise returns 0
export const perceptronActivation: ActivationFunction = (x: number) => {
    return x > 0 ? 1 : 0;
};

export const sigmoid: ActivationFunction = (x: number) => {
    return 1 / (1 + Math.exp(-x));
};


export const sigmoidDerivative: ActivationFunction = (activationOutput: number) => {
    return activationOutput * (1 - activationOutput);
};
