require('@tensorflow/tfjs-node')
const tf = require('@tensorflow/tfjs')
const loadCSV = require('../load-csv')
const LogisticRegression = require('./logistic-regression')
const plot = require('node-remote-plot')

const { features, labels, testFeatures, testLabels } = loadCSV(
  '../data/cars.csv',
  {
    dataColumns: ['horsepower', 'displacement', 'weight'],
    labelColumns: ['passedemissions'],
    shuffle: true,
    splitTest: 50,
    converters: {
      passedemissions(value) {
        return value === 'TRUE' ? 1 : 0
      },
    },
  },
)

// console.log(labels)

const regression = new LogisticRegression(features, labels, {
  learningRate: 0.5, // random guess, will be optimized later
  iterations: 100,
  batchSize: 50,
  decisionBoundary: 0.5,
})

regression.train()
regression.predict([[88, 97, 1.065]]).print()

// console.log(regression.test(testFeatures, testLabels))

console.log(regression.test(testFeatures, testLabels))

plot({
  x: regression.costHistory.reverse(),
})
