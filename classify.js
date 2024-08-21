import * as tf from '@tensorflow/tfjs-node';
import fs from 'fs';
import path from 'path';
import { MongoClient } from 'mongodb';
import { fileURLToPath } from 'url';

const MOBILE_NET_INPUT_HEIGHT = 224;
const MOBILE_NET_INPUT_WIDTH = 224;
const MONGO_URI = 'mongodb://127.0.0.1:27017'; // MongoDB URI
const DATABASE_NAME = 'imageRetrievalDB';
const COLLECTION_NAME = 'imageFeatures';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const imageDir = path.join(__dirname, 'images'); // Đường dẫn tới thư mục chứa hình ảnh
const labelsFilePath = path.join(__dirname, 'models/label.json'); // Đường dẫn tới file label.json

let model, baseModel;

async function loadMobileNetFeatureModel() {
  const url = 'https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/SavedModels/mobilenet-v2/model.json';
  model = await tf.loadLayersModel(url);
  console.log('MobileNetV2 model loaded successfully!');

  const layer = model.getLayer('global_average_pooling2d_1'); // Xác nhận tên layer
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
    let normalizedTensorImage = resizedTensorImage.div(255.0);
    return normalizedTensorImage.expandDims(0);
  });
}

async function extractFeature(image) {
  return tf.tidy(() => {
    let processedImage = preprocess(image);
    let imageFeatures = baseModel.predict(processedImage);
    let featuresArray = imageFeatures.squeeze().arraySync(); // Chuyển đổi tensor thành mảng số
    console.log('Extracted Features:', featuresArray); // Debug log
    return featuresArray;
  });
}

async function saveFeaturesToMongoDB(features, label) {
  const client = new MongoClient(MONGO_URI);
  try {
    await client.connect();
    const db = client.db(DATABASE_NAME);
    const collection = db.collection(COLLECTION_NAME);

    await collection.insertOne({ label, features });
    console.log('Features saved to MongoDB.');
  } finally {
    await client.close();
  }
}

async function processImages() {
  await loadMobileNetFeatureModel();

  // Đọc nhãn lớp từ file label.json
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
        let features = await extractFeature(imageBase64); // Chờ kết quả

        // Lưu vector vào MongoDB với nhãn tương ứng
        let label = folder.name;
        await saveFeaturesToMongoDB(features, label);
      }
    }
  }
}
function cosineSimilarity(tensorA, tensorB) {
  // Tính toán dot product
  const dotProduct = tf.tidy(() => tf.sum(tf.mul(tensorA, tensorB)).arraySync());
  
  // Tính toán norm (chuẩn) của mỗi tensor
  const normA = tf.tidy(() => tf.norm(tensorA).arraySync());
  const normB = tf.tidy(() => tf.norm(tensorB).arraySync());
  
  // Tính toán cosine similarity
  return dotProduct / (normA * normB);
}

async function classifyImage(imagePath) {
  await loadMobileNetFeatureModel();

  const absoluteImagePath = path.resolve(imagePath);
  console.log('Checking file path:', absoluteImagePath);

  if (!fs.existsSync(absoluteImagePath)) {
    console.error('File does not exist:', absoluteImagePath);
    return { productName: 'Not Determined', confidence: 0 };
  }

  const imageBuffer = fs.readFileSync(absoluteImagePath);
  const imageBase64 = `data:image/jpeg;base64,${imageBuffer.toString('base64')}`;
  const features = await extractFeature(imageBase64);

  const client = new MongoClient(MONGO_URI);
  try {
    await client.connect();
    const db = client.db(DATABASE_NAME);
    const collection = db.collection(COLLECTION_NAME);

    const allImages = await collection.find().toArray();

    if (allImages.length === 0) {
      console.error('No images found in the database.');
      return { productName: 'Not Determined', confidence: 0 };
    }

    const similarImages = allImages.map(image => {
      const featureTensor = tf.tensor(image.features);
      const featuresTensor = tf.tensor(features);

      if (featureTensor.shape[0] !== featuresTensor.shape[0]) {
        console.error('Tensor shape mismatch:', featureTensor.shape, featuresTensor.shape);
        return { ...image, distance: Infinity };
      }

      // Sử dụng cosine similarity
      const similarity = cosineSimilarity(featuresTensor, featureTensor);
      const distance = 1 - similarity; // Chuyển đổi thành khoảng cách
      return { ...image, distance };
    });

    similarImages.sort((a, b) => a.distance - b.distance);

    const bestPrediction = similarImages[0];
    const distance = bestPrediction ? bestPrediction.distance : 1;

    // Điều chỉnh ngưỡng confidence
    const confidence = isNaN(distance) ? 0 : Math.max(0, 100 - (distance * 100));
    if (confidence >= 50) {
      const productName = confidence >= 60 ? bestPrediction.label : 'Not Determined'; // Thay đổi ngưỡng ở đây
      console.log('Product Name:', productName);
      console.log('Confidence:', confidence.toFixed(2), '%');
      return { productName, confidence };
    } else {
      console.log('Confidence is below 50%, not displaying result.');
      console.log('Confidence:', confidence.toFixed(2), '%');
      return { productName: 'Not Determined', confidence };
    }
  } finally {
    await client.close();
  }
}

// Ví dụ sử dụng
const testImagePath = 'D:/Download/anhbinhhoa.jpg';
classifyImage(testImagePath).then(result => {
  console.log('Result:', result);
}).catch(err => {
  console.error('Error:', err);
});
