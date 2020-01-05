const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LinearRegression {
  constructor(features, labels, options) {
    this.features = tf.tensor(features)
    this.labels = tf.tensor(labels)
    this.features = tf
      .ones([this.features.shape[0], 1])
      .concat(this.features, 1) // number of columns will always be 1, rows will be same as rows in features
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options,
    )
    this.weights = tf.zeros([2, 1]) // there are only 2 weights, m and b
  }

  // gradientDescent() {
  //   const currentGuessesForMPG = this.features.map(row => {
  //     return this.m * row[0] + this.b // these are current guesses for MPG values
  //   })

  //   const bSlope =
  //     (_.sum(
  //       currentGuessesForMPG.map((guess, i) => {
  //         return guess - this.labels[i][0] // i is same index as the guess
  //       }),
  //     ) *
  //       2) /
  //     this.features.length

  //   const mSlope =
  //     (_.sum(
  //       currentGuessesForMPG.map((guess, i) => {
  //         return -1 * this.features[i][0] * (this.labels[i][0] - guess)
  //       }),
  //     ) *
  //       2) /
  //     this.features.length

  //   this.m = this.m - mSlope * this.options.learningRate
  //   this.b = this.b - bSlope * this.options.learningRate
  // }

  gradientDescent() {
    const currentGuesses = this.features.matMul(this.weights) // this will do matrix multiplication, not elementwise multiplication
    const differences = currentGuesses.sub(this.labels)

    const slopes = this.features
      .transpose()
      .matMul(differences)
      .div(this.features.shape[0]) // shape[0] because the shape is row columns. and we need the number of rows. You could also do a mul(2) per equation, but since the learning rate will mod anyway, we can leave that off

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate)) // this will modify the weights so we can try to get to that zero slope
  }

  train() {
    for (let i = 0; i < this.options.iterations; i++) {
      this.gradientDescent()
    }
  }

  test(testFeatures, testLabels) {
    testFeatures = tf.tensor(testFeatures)
    testLabels = tf.tensor(testLabels)

    testFeatures = tf.ones([testFeatures.shape[0], 1]).concat(testFeatures, 1)
    const predictions = testFeatures.matMul(this.weights)
    const sumSquaresResiduals = testLabels
      .sub(predictions)
      .pow(2)
      .sum() // sum of all the values, so we don't need to provide an axis
      .get() // don't have to pass arguments to get because there is 1 number inside the tensor

    const sumSquaresTotals = testLabels
      .sub(testLabels.mean()) // because for totals we want to subtract mean each time
      .pow(2)
      .sum()
      .get()

    return 1 - sumSquaresResiduals / sumSquaresTotals // this is the coefficient of determination calculation
    // predictions.print()
  }
}

module.exports = LinearRegression
