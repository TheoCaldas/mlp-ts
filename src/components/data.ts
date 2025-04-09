import fs from "fs";
import Rand from 'rand-seed';

// Each item has x and y coordinates and a color label
export type Item = {x: number, y: number, color: string};

// Read a JSON file and parse it into an array of items
export const parseJSON = (filePath: string): Item[] => {
  return JSON.parse(fs.readFileSync(filePath, "utf-8")) as Item[];
}

// Extract input and label arrays from an array of items
export const extractInputLabel = (labeledData: Item[]): {data: number[][], labels: number[]} => { 
  const data = labeledData.map(item => [item.x, item.y]);
  const labels = labeledData.map(item => (item.color == 'a') ? 1 : 0);
  return { data, labels };
}

// Normalize x and y data values to be between 0 and 1
export const normalize = (rawData: Item[]): Item[] => {
  const minX = Math.min(...rawData.map(item => item.x));
  const maxX = Math.max(...rawData.map(item => item.x));
  const minY = Math.min(...rawData.map(item => item.y));
  const maxY = Math.max(...rawData.map(item => item.y));
  const rangeX = maxX - minX;
  const rangeY = maxY - minY;

  // console.log(minX, maxX, minY, maxY);
  return rawData.map(item => ({
    x: (item.x - minX) / rangeX,
    y: (item.y - minY) / rangeY,
    color: item.color,
  }));
}

// Shuffle the data array randomly
export const shuffle = (data: Item[]): Item[] => {
  const shuffled = data.slice();
  return shuffled.sort(() => Math.random() - 0.5);
}

// Shuffle the data array given a seed
export const fixedShuffle = (data: Item[], seed: string): Item[] => {
  const rand = new Rand(seed);
  const shuffled = data.slice();
  return shuffled.sort(() => rand.next() - 0.5);
}

// Split the data into training and testing sets based on a ratio
// The ratio is the proportion to the amount of data to be used for training
export const split = (data: Item[], ratio: number): {trainData: Item[], testData: Item[]} => {
  const splitIndex = Math.floor(data.length * ratio);
  const trainData = data.slice(0, splitIndex);
  const testData = data.slice(splitIndex);
  return { trainData: trainData, testData: testData };
}