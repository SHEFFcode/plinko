require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('./load-csv')
const LinearRegression = require('./linear-regression')

let { features, labels, testFeatures, testLabels } = loadCSV('./cars.csv', {
  shuffle: true,
  splitTest: 50,
  dataColumns: ['horsepower', 'weight', 'displacement'],
  labelColumns: ['mpg'],
})

const regression = new LinearRegression(features, labels, {
  learningRate: 0.1, // generally small learningRate will mean that we will have to do a ton of iterations
  iterations: 100,
})

// regression.features.print()

regression.train()
const r2 = regression.test(testFeatures, testLabels)
console.log(`R2 is ${r2}`)
