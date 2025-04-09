import { parseJSON, extractInputLabel, normalize, shuffle, split, Item, fixedShuffle } from "./data";
import { describe, it, expect } from 'vitest';

describe("Data module", () => {
    const mockData: Item[] = [
        { x: 1, y: 2, color: "a" },
        { x: 3, y: 4, color: "b" },
        { x: 5, y: 6, color: "a" },
        { x: 7, y: 8, color: "b" },
    ];

    describe("parseJSON", () => {
        it("should parse JSON file into an array of items", () => {
            const filePath = "src/datasets/points/mock.json";
            const result = parseJSON(filePath);
            expect(result).toEqual(mockData);
        });
    });

    describe("extractInputLabel", () => {
        it("should extract input and label arrays from labeled data", () => {
            const { data, labels } = extractInputLabel(mockData);

            expect(data).toEqual([
                [1, 2],
                [3, 4],
                [5, 6],
                [7, 8],
            ]);
            expect(labels).toEqual([1, 0, 1, 0]);
        });
    });

    describe("normalize", () => {
        it("should normalize x and y values to be between 0 and 1", () => {
            const normalizedData = normalize(mockData);

            expect(normalizedData).toEqual([
                { x: 0, y: 0, color: "a" },
                { x: 0.3333333333333333, y: 0.3333333333333333, color: "b" },
                { x: 0.6666666666666666, y: 0.6666666666666666, color: "a" },
                { x: 1, y: 1, color: "b" },
            ]);
        });
    });

    describe("shuffle", () => {
        it("should shuffle the data array randomly", () => {
            const shuffledData = shuffle(mockData);
            expect(shuffledData).toHaveLength(mockData.length);
            expect(shuffledData).toEqual(expect.arrayContaining(mockData));
        });
    });

    describe("fixedShuffle", () => {
        it("should shuffle the data array with a fixed seed", () => {
            const seed = "123";
            const fixedShuffledData1 = fixedShuffle(mockData, seed);
            const fixedShuffledData2 = fixedShuffle(mockData, seed);

            expect(mockData).not.toEqual(fixedShuffledData1);
            expect(fixedShuffledData1).toEqual(fixedShuffledData2);

            expect(fixedShuffledData1).toHaveLength(mockData.length);
            expect(fixedShuffledData1).toEqual(expect.arrayContaining(mockData));
        });
    });

    describe("split", () => {
        it("should split data into training and testing sets based on the ratio", () => {
            const ratio = 0.5;
            const { trainData, testData } = split(mockData, ratio);

            expect(trainData).toHaveLength(2);
            expect(testData).toHaveLength(2);
            expect([...trainData, ...testData]).toEqual(mockData);
        });
    });
});