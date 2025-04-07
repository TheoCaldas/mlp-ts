export type ActivationFunction = (x: number) => number;

export type Perceptron = {
    inputs?: number[],
    weights: number[],
    bias: number,
    output?: number,
    activationFunc: ActivationFunction,
}

export const dotProduct = (a: number[], b: number[]) => {
    if(a.length !== b.length) {
        throw new Error("Arrays must be of the same length");
    }
    let sum = 0;
    for(let i = 0; i < a.length; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

export const init = (weights: number[], bias: number, func: ActivationFunction): Perceptron =>  {
    const perceptron = {
        weights: weights,
        bias: bias,
        activationFunc: func,
    };
    return perceptron;
}

export const initRandom = (nInputs: number, func: ActivationFunction): Perceptron =>  {
    const weights = Array.from({ length: nInputs }, () => Math.random());
    const bias = Math.random();
    return init(weights, bias, func);
}

export const updateOutput = (perceptron: Perceptron, inputs: number[]): Perceptron => {
    if (inputs.length !== perceptron.weights.length) {
        throw new Error("Input length must match weights length");
    }
    perceptron.inputs = inputs;
    const zValue = dotProduct(perceptron.inputs, perceptron.weights) + perceptron.bias;
    perceptron.output = perceptron.activationFunc(zValue);

    return perceptron;
}