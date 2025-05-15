import { parseJSON, normalize, shuffle, split, extractInputLabel, fixedShuffle } from "./components/data";
import { exportModelSLP, importModelSLP, initTrainingSLP, testSLP, trainSLP } from "./components/slp/slp.train";

// Hyperparameters
const dataFilePath = "src/datasets/points/data3.json";
const modelFilePath = "src/models/slp/slp3.2.json";
const splitRatio = 0.8;
const learningRate = 0.1;
const hiddenLayerSize = 10;

// Handle dataset
const data = parseJSON(dataFilePath);
const normalizedData = normalize(data);
// const shuffledData = shuffle(normalizedData);
const shuffledData = fixedShuffle(normalizedData, '123');
const { trainData, testData } = split(shuffledData, splitRatio);
const { data: trainInputs, labels: trainLabels } = extractInputLabel(trainData);
const { data: testInputs, labels: testLabels } = extractInputLabel(testData);

// Train model
let model = initTrainingSLP(trainInputs, trainLabels, hiddenLayerSize, learningRate);
model = trainSLP(model, 1000);

// OR Import model
// let model = importModelSLP(modelFilePath);

// Test model
let accuracy = testSLP(model, testInputs, testLabels, false);
console.log(`\nAccuracy: ${(accuracy * 100).toFixed(2)}%`);

// Export model
exportModelSLP(model, modelFilePath);