import { initTraining, test, train } from "./components/perceptron.train";
import { parseJSON, normalize, shuffle, split, extractInputLabel } from "./components/data";

// Hyperparameters
const filePath = "src/datasets/points/data1.json";
const splitRatio = 0.8;
const learningRate = 0.1;

// Handle dataset
const data = parseJSON(filePath);
const normalizedData = normalize(data);
const shuffledData = shuffle(normalizedData);
const { trainData, testData } = split(shuffledData, splitRatio);
const { data: trainInputs, labels: trainLabels } = extractInputLabel(trainData);
const { data: testInputs, labels: testLabels } = extractInputLabel(testData);

// Train the model
let model = initTraining(trainInputs, trainLabels, learningRate);
model = train(model);

// Test the model
let accuracy = test(model, testInputs, testLabels);
console.log(`\nAccuracy: ${(accuracy * 100).toFixed(2)}%`);