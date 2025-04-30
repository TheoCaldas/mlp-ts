import { exportModel, importModel, initTraining, test, train } from "./components/perceptron/perceptron.train";
import { parseJSON, normalize, shuffle, split, extractInputLabel, fixedShuffle } from "./components/data";

// Hyperparameters
const dataFilePath = "src/datasets/points/data3.json";
const modelFilePath = "src/models/perceptron3.json";
const splitRatio = 0.8;
const learningRate = 0.1;

// Handle dataset
const data = parseJSON(dataFilePath);
const normalizedData = normalize(data);
// const shuffledData = shuffle(normalizedData);
const shuffledData = fixedShuffle(normalizedData, '123');
const { trainData, testData } = split(shuffledData, splitRatio);
const { data: trainInputs, labels: trainLabels } = extractInputLabel(trainData);
const { data: testInputs, labels: testLabels } = extractInputLabel(testData);

// Train model
// let model = initTraining(trainInputs, trainLabels, learningRate);
// model = train(model);

// OR Import model
let model = importModel(modelFilePath);

// Test model
let accuracy = test(model, testInputs, testLabels);
console.log(`\nAccuracy: ${(accuracy * 100).toFixed(2)}%`);

// Export model
// exportModel(model, modelFilePath);