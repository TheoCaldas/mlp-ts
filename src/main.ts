import { initTraining, test, train } from "./components/perceptron.train";

const data = [
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],

  [0.1, 0.2],
  [0.1, 0.9],
  [0.9, 0.2],
  [0.9, 0.99],

  [0.01, 0.3],
  [0.3, 0.6],
  [0.8, 0.1],
  [0.6, 0.7],

  [0.4, 0.1],
  [0.3, 0.8],
  [0.7, 0],
  [1, 0.6],

  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],

  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1],
];
const labels = [
  0, 0, 0, 1,
  0, 0, 0, 1,
  0, 0, 0, 1,
  0, 0, 0, 1,
  0, 0, 0, 1,
  0, 0, 0, 1,
]; 
const learningRate = 0.1;

let model = initTraining(data, labels, learningRate);
model = train(model);

let _ = test(model, data, labels);