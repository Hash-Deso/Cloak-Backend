const express = require('express');
const axios = require('axios').default;
require('dotenv').config({ path: './config.env' });
const cors = require('cors');
const tf = require('@tensorflow/tfjs-node');
const nsfw = require('nsfwjs');
const sharp = require('sharp');
let model = null;

const app = express();
app.use(
  cors({
    origin: '*',
    credentials: true,
  })
);
app.use(express.json());

app.post('/detect', async (req, res, next) => {
  try {
    const { images } = req.body;
    images.shift();
    if (!images) {
      throw new Error('Bad Request');
    }
    const result = [];
    if (!model) {
      console.log('Loading model');
      model = await nsfw.load();
    }
    for (let i = 0; i < images.length; i++) {
      const pic = await axios.get(images[i], {
        responseType: 'arraybuffer',
      });
      const jpegBuffer = await sharp(pic.data).jpeg().toBuffer();
      const image = tf.node.decodeImage(jpegBuffer, 3);
      const predictions = await model.classify(image);
      const importantClasses = { Hentai: 10, Porn: 2, Sexy: 80 };
      for (let j = 0; j < predictions.length; j++) {
        const prediction = predictions[j];
        if (
          importantClasses.hasOwnProperty(prediction.className) &&
          prediction.probability * 100 >= importantClasses[prediction.className]
        ) {
          result.push(images[i]);
        }
      }
      image.dispose();
    }
    res.status(200).json({ status: 'success', result });
  } catch (e) {
    next(e);
  }
});

app.use((err, req, res, next) => {
  res.status(500).json({ status: 'error', error: err.message });
});
const PORT = process.env.PORT || 3000;
nsfw.load().then((main) => {
  model = main;
  app.listen(PORT, () => {
    console.log('Server Started');
  });
});
