import * as tf from "@tensorflow/tfjs-node";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const MOBILE_NET_INPUT_HEIGHT = 224;
const MOBILE_NET_INPUT_WIDTH = 224;
const BATCH_SIZE = 4;
const EPOCH = 50;
const LEARNING_RATE = 0.001;
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const imageDir = path.join(__dirname, "images");

let model, baseModel, customModel, combinedModel, localModel;
let traningDataInputs = [],
  trainingDataOutputs = [];

const classFolders = await fs.promises.readdir(imageDir, {
  withFileTypes: true,
});

const labels = classFolders.map((classData) => classData.name);
const jsonString = JSON.stringify(labels, null, 2);

const outputPath = path.join('./models', "label.json");
fs.writeFile(outputPath, jsonString, (err) => {
  if (err) {
    console.error("Error writing file:", err);
  } else {
    console.log("Label file has been saved.");
  }
});

// function
//load model
async function loadMobileNetFeatureModel() {
  const url =
    "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json";
  model = await tf.loadLayersModel(url);
  console.log("MobileNet v2 loaded successfully!");

  const layer = model.getLayer("global_average_pooling2d_1");
  baseModel = tf.model({ inputs: model.inputs, outputs: layer.output });
  // baseModel.summary()

  tf.tidy(function () {
    let answer = baseModel.predict(
      tf.zeros([1, MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH, 3])
    );
    console.log(answer.shape);
  });
}

//precessor (base64 to )
function preprocess(image) {
  return tf.tidy(() => {
    // Ensure that 'image' is a string
    if (typeof image !== "string") {
      throw new TypeError(
        "Expected image to be a base64 string, but received " + typeof image
      );
    }

    let base64Image = image.replace(/^data:image\/(png|jpeg);base64,/, "");
    const buffer = Buffer.from(base64Image, "base64");
    let imageAsTensor = tf.node.decodeImage(buffer, 3);
    let resizedTensorImage = tf.image.resizeBilinear(
      imageAsTensor,
      [MOBILE_NET_INPUT_HEIGHT, MOBILE_NET_INPUT_WIDTH],
      true
    );
    let normalizedTensorImage = resizedTensorImage.div(255);
    return normalizedTensorImage.expandDims();
  });
}

//extract features
function extractFeature(image) {
  let processedImage = preprocess(image);
  return tf.tidy(() => {
    let imageFeatures = baseModel.predict(processedImage);
    return imageFeatures.squeeze();
  });
}

//train
async function train() {
  for (let folder of classFolders) {
    if (folder.isDirectory()) {
      let folderPath = path.join(imageDir, folder.name);
      let images = await fs.promises.readdir(folderPath);

      for (let imageFile of images) {
        let imagePath = path.join(folderPath, imageFile);
        let imageBuffer = fs.readFileSync(imagePath);
        let imageBase64 = `data:image/jpeg;base64,${imageBuffer.toString(
          "base64"
        )}`;
        let features = extractFeature(imageBase64); // Ensure to await here

        traningDataInputs.push(features);
        console.log("input: " + features);
        trainingDataOutputs.push(labels.indexOf(folder.name));
        console.log("output: " + labels.indexOf(folder.name));
      }
    }
  }

  // Stack inputs, convert outputs to tensor, and perform training as before
  tf.util.shuffleCombo(traningDataInputs, trainingDataOutputs);
  let inputsAsTensor = tf.stack(traningDataInputs);
  let outputsAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
  let oneHotOutputs = tf.oneHot(outputsAsTensor, labels.length);

  await customModel.fit(inputsAsTensor, oneHotOutputs, {
    batchSize: BATCH_SIZE,
    validationSplit: 0.1,
    epochs: EPOCH,
    callbacks: [
      {
        onEpochEnd: (epoch, logs) =>
          console.log("Data for epoch " + epoch + 1, logs),
      },
      //   tf.callbacks.earlyStopping({
      //     monitor: 'val_loss',
      //     patience: 10  // Stops training if val_loss does not improve for 10 consecutive epochs
      // })
    ],
  });

  inputsAsTensor.dispose();
  outputsAsTensor.dispose();
  oneHotOutputs.dispose();

  combinedModel = tf.sequential();
  combinedModel.add(baseModel);
  combinedModel.add(customModel);
  combinedModel.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
  });
  //   combinedModel.summary();
  await combinedModel.save("file://./models");
}

await loadMobileNetFeatureModel();

//add custom layer
customModel = tf.sequential();
customModel.add(
  tf.layers.dense({ inputShape: [1280], units: 128, activation: "relu" })
);
customModel.add(
  tf.layers.dense({ units: labels.length, activation: "softmax" })
);
const optimizer = tf.train.adam(LEARNING_RATE);
customModel.compile({
  optimizer: optimizer,
  loss: "categoricalCrossentropy",
  metrics: ["accuracy"],
});
customModel.summary();

await train();
