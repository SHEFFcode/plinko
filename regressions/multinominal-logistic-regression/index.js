require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('../load-csv')
const LogisticRegression = require('./multivariate-logistic-regression')
const plot = require('node-remote-plot')
const _ = require('lodash')

const { features, labels, testFeatures, testLabels } = loadCSV(
  '../data/cars.csv',
  {
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['mpg'],
    shuffle: true,
    splitTest: 50,
    converters: {
      mpg(value) {
        const mpg = parseFloat(value)
        if (mpg < 15) {
          return [1, 0, 0] // fuel efficiency tensor: [low, medium, high]
        } else if (mpg < 30) {
          return [0, 1, 0]
        } else {
          return [0, 0, 1]
        }
      },
    },
  },
)

// console.log(_.flatMap(labels)) // flattens and removes one layer of nesting

const regression = new LogisticRegression(features, _.flatMap(labels), {
  learningRate: 0.5, // random guess, will be optimized later
  iterations: 100,
  batchSize: 10,
})

// regression.weights.print()

regression.train()
// regression.predict([[150, 200, 2.223]]).print()
console.log(regression.test(testFeatures, _.flatMap(testLabels))) // accuracy percentage value
