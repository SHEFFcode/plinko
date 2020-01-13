require('@tensorflow/tfjs-node')
const LogisticRegression = require('./multivariate-logistic-regression')
const _ = require('lodash')
const mnist = require('mnist-data')

const mnistData = mnist.training(0, 5000) // give me all the training observations from 0 to 5000, not inclusive
const features = mnistData.images.values.map(image => _.flatMap(image))

const encodedLabels = mnistData.labels.values.map(label => {
  const row = new Array(10).fill(0)
  row[label] = 1
  return row
})

const regression = new LogisticRegression(features, encodedLabels, {
  learningRate: 1,
  iterations: 20,
  batchSize: 100,
})

regression.train()

const testMnistData = mnist.testing(0, 100) // let's load up first 100 images from training data set

const testFeatures = testMnistData.images.values.map(image => _.flatMap(image))
const testEncodedLabels = testMnistData.labels.values.map(label => {
  const row = new Array(10).fill(0)
  row[label] = 1
  return row
})

const accuracy = regression.test(testFeatures, testEncodedLabels)
console.log(`Accuracy is ${accuracy}`)
