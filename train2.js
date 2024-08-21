import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import path from 'path';
import { MongoClient } from 'mongodb';
import { fileURLToPath } from 'url';

const MOBILE_NET_INPUT_HEIGHT = 224;
const MOBILE_NET_INPUT_WIDTH = 224;
const MONGO_URI = 'mongodb://thanhnguyen:1Thanh23456%40@34.81.6.236:27017';
// const MONGO_URI = 'mongodb://127.0.0.1:27017'; // MongoDB URI
const DATABASE_NAME = 'imageRetrievalDB';
const COLLECTION_NAME = 'imageFeatures';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const imageDir = path.join(__dirname, 'images');
const labelsFilePath = path.join(__dirname, 'models/label.json');

let model, baseModel;

async function loadMobileNetFeatureModel() {
  const url = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json';
  model = await tf.loadLayersModel(url);
  console.log('MobileNetV2 model loaded successfully!');

  const layer = model.getLayer('global_average_pooling2d_1');
  baseModel = tf.model({ inputs: model.inputs, outputs: layer.output });
  baseModel.summary();
}

function preprocess(image) {
  return tf.tidy(() => {
    if (typeof image !== 'string') {
      throw new TypeError('Expected image to be a base64 string.');
    }

    let base64Image = image.replace(/^data:image\/(png|jpeg);base64,/, '');
    const buffer = Buffer.from(base64Image, 'base64');
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

function extractFeature(image) {
  let processedImage = preprocess(image);
  return tf.tidy(() => {
    let imageFeatures = baseModel.predict(processedImage);
    return imageFeatures.squeeze();
  });
}

async function saveFeaturesToMongoDB(features, label) {
  const client = new MongoClient(MONGO_URI);
  try {
    await client.connect();
    const db = client.db(DATABASE_NAME);
    const collection = db.collection(COLLECTION_NAME);
    
    await collection.insertOne({ label, features: features.arraySync() });
    console.log('Features saved to MongoDB.');
  } finally {
    await client.close();
  }
}

async function processImages() {
  await loadMobileNetFeatureModel();

  const labels = JSON.parse(fs.readFileSync(labelsFilePath, 'utf8'));
  console.log('Labels:', labels);

  const classFolders = await fs.promises.readdir(imageDir, { withFileTypes: true });

  for (let folder of classFolders) {
    if (folder.isDirectory()) {
      let folderPath = path.join(imageDir, folder.name);
      let images = await fs.promises.readdir(folderPath);

      for (let imageFile of images) {
        let imagePath = path.join(folderPath, imageFile);
        let imageBuffer = fs.readFileSync(imagePath);
        let imageBase64 = `data:image/jpeg;base64,${imageBuffer.toString('base64')}`;
        let features = extractFeature(imageBase64);
        
        let label = folder.name;
        await saveFeaturesToMongoDB(features, label);
      }
    }
  }
}

processImages();
