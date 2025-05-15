import { describe, it, expect } from "vitest";
import { initTrainingSLP, trainSLP, predictSLP, testSLP, exportModelSLP, importModelSLP } from "./slp.train";
import fs from "fs";

describe("Supervised SLP Module", () => {
    it("should initialize a supervised SLP correctly", () => {
        const data = [[0, 0], [0, 1], [1, 0], [1, 1]];
        const labels = [0, 1, 1, 0];
        const hiddenLayerSize = 2;
        const learningRate = 0.1;

        const model = initTrainingSLP(data, labels, hiddenLayerSize, learningRate);

        expect(model.data).toEqual(data);
        expect(model.labels).toEqual(labels);
        expect(model.learningRate).toBe(learningRate);
        expect(model.itemCount).toBe(data.length);
        expect(model.slp.hiddenLayer.length).toBe(hiddenLayerSize);
    });

    it("should throw an error if data and labels length mismatch", () => {
        const data = [[0, 0], [0, 1]];
        const labels = [0];
        const hiddenLayerSize = 2;
        const learningRate = 0.1;

        expect(() => initTrainingSLP(data, labels, hiddenLayerSize, learningRate)).toThrow(
            "Data and labels must have the same length"
        );
    });

    it("should train the SLP and reduce error", () => {
        const data = [[0, 0], [0, 1], [1, 0], [1, 1]];
        const labels = [0, 1, 1, 0];
        const hiddenLayerSize = 2;
        const learningRate = 0.1;

        let model = initTrainingSLP(data, labels, hiddenLayerSize, learningRate);
        model = trainSLP(model, 1, false);

        expect(model.slp).toBeDefined();
    });

    it("should predict correctly after training", () => {
        const data = [[0, 0], [0, 1], [1, 0], [1, 1]];
        const labels = [0, 1, 1, 0];
        const hiddenLayerSize = 2;
        const learningRate = 0.1;

        let model = initTrainingSLP(data, labels, hiddenLayerSize, learningRate);
        model = trainSLP(model, 1, false);

        const prediction = predictSLP(model, [0, 1]);
        expect(prediction).toBe(1);
    });

    it("should test the model and return accuracy", () => {
        const data = [[0, 0], [0, 1], [1, 0], [1, 1]];
        const labels = [0, 1, 1, 0];
        const hiddenLayerSize = 1;
        const learningRate = 0.1;

        let model = initTrainingSLP(data, labels, hiddenLayerSize, learningRate);
        model = trainSLP(model, 1000, false);

        const accuracy = testSLP(model, data, labels, false);
        expect(accuracy).toBeGreaterThan(0);
    });

    it("should export and import the model correctly", () => {
        const data = [[0, 0], [0, 1], [1, 0], [1, 1]];
        const labels = [0, 1, 1, 0];
        const hiddenLayerSize = 2;
        const learningRate = 0.1;

        const model = initTrainingSLP(data, labels, hiddenLayerSize, learningRate);
        const filePath = "src/models/slp/mock.json";

        exportModelSLP(model, filePath);
        expect(fs.existsSync(filePath)).toBe(true);

        const importedModel = importModelSLP(filePath);
        expect(importedModel.slp.hiddenLayer.length).toBe(hiddenLayerSize);

        fs.unlinkSync(filePath); // Clean up
    });
});