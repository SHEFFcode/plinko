require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('../load-csv')
const LinearRegression = require('./linear-regression')
const plot = require('node-remote-plot')

let { features, labels, testFeatures, testLabels } = loadCSV(
  '../data/cars.csv',
  {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg'],
  },
)

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1, // generally small learningRate will mean that we will have to do a ton of iterations
  iterations: 3, // with batch gradient descent I can lower this value from 100 to 3!
  batchSize: 10,
})

// regression.features.print()

regression.train()
const r2 = regression.test(testFeatures, testLabels)

plot({
  x: regression.mseHistory,
  xLabel: 'Iteration #',
  yLabel: 'Mean Squared Error',
})

regression.predict([[120, 2, 380]]).print()

console.log(`R2 is ${r2}`)
