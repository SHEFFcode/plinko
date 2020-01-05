const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LinearRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features)
    this.labels = tf.tensor(labels)
    this.mseHistory = []
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000 },
      options,
    )
    this.weights = tf.zeros([this.features.shape[1], 1]) // the number of rows of weights will be equal to the number of columns in features (this allows for matrix multiplication)
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
      this.recordMSE()
      this.updateLearningRate()
    }
  }

  test(testFeatures, testLabels) {
    testFeatures = this.processFeatures(testFeatures)
    testLabels = tf.tensor(testLabels)

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

  processFeatures(features) {
    let tfFeatures = tf.tensor(features)
    let standardizedTFFeatures = this.standardize(tfFeatures)
    tfFeatures = tf
      .ones([standardizedTFFeatures.shape[0], 1])
      .concat(standardizedTFFeatures, 1) // ones need to happen after standardization, otherwise we will standardize the 1s which will screw them up

    return tfFeatures
  }

  standardize(features) {
    if (!this.mean && !this.variance) {
      const { mean, variance } = tf.moments(features, 0) // this is something that tensorflow is able to produce out of the box for us
      this.mean = mean
      this.variance = variance
    }

    return features.sub(this.mean).div(this.variance.pow(0.5)) // standardization formula.
  }

  recordMSE() {
    const mse = this.features
      .matMul(this.weights)
      .sub(this.labels)
      .pow(2)
      .sum()
      .div(this.features.shape[0])
      .get() // so that we get a number and not a tensor

    this.mseHistory.push(mse)
  }

  updateLearningRate() {
    if (this.mseHistory.length >= 2) {
      const lastValue = this.mseHistory[this.mseHistory.length - 1]
      const secondToLastValue = this.mseHistory[this.mseHistory.length - 2]

      if (lastValue > secondToLastValue) {
        // we are going in the wrong direction
        this.options.learningRate /= 2
      } else {
        this.options.learningRate *= 1.05
      }
    }
  }
}

module.exports = LinearRegression
