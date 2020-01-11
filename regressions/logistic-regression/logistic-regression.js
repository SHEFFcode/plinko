const tf = require('@tensorflow/tfjs')
const _ = require('lodash')

class LogisticRegression {
  constructor(features, labels, options) {
    this.features = this.processFeatures(features)
    this.labels = tf.tensor(labels)
    this.mseHistory = []
    this.bHistory = [] // all the diff values of b that we attempt to define relationship between car attributes and car MPG
    this.options = Object.assign(
      { learningRate: 0.1, iterations: 1000, decisionBoundary: 0.5 },
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

  gradientDescent(features, labels) {
    const currentGuesses = features.matMul(this.weights).sigmoid() // this will do matrix multiplication, not elementwise multiplication
    const differences = currentGuesses.sub(labels)

    const slopes = features
      .transpose()
      .matMul(differences)
      .div(features.shape[0]) // shape[0] because the shape is row columns. and we need the number of rows. You could also do a mul(2) per equation, but since the learning rate will mod anyway, we can leave that off

    this.weights = this.weights.sub(slopes.mul(this.options.learningRate)) // this will modify the weights so we can try to get to that zero slope
  }

  train() {
    const batchQuantity = Math.floor(
      this.features.shape[0] / this.options.batchSize,
    )

    for (let i = 0; i < this.options.iterations; i++) {
      for (let j = 0; j < batchQuantity; j++) {
        const { batchSize } = this.options
        const startIndex = j * batchSize

        const featureSlice = this.features.slice(
          [startIndex, 0],
          [batchSize, -1],
        ) // first batch of features to run gradient descent with

        const labelSlice = this.labels.slice([startIndex, 0], [batchSize, -1])

        this.gradientDescent(featureSlice, labelSlice)
      }

      // Update the values here after going through a full batch
      this.bHistory.push(this.weights.get(0, 0)) // first element in the weights tensor is the b value
      this.recordMSE()
      this.updateLearningRate()
    }
  }

  /**
   * Method that takes in an array of arrays of car observations
   * and returns a prediction for its MPG rating.
   * @param {array<array<number>>} observations
   * @returns {tensor} mpgEstimate
   */
  predict(observations) {
    return (
      this.processFeatures(observations)
        .matMul(this.weights)
        .sigmoid()
        // .round() // this will round up or down at .5 decision margin
        .greater(this.options.decisionBoundary)
        .cast('float32')
    ) // another way to round, but u can specify the value to round on
  }

  test(testFeatures, testLabels) {
    const predictions = this.predict(testFeatures).round()
    testLabels = tf.tensor(testLabels)

    const incorrect = predictions
      .sub(testLabels)
      .abs()
      .sum()
      .get()

    return (predictions.shape[0] - incorrect) / predictions.shape[0]
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

module.exports = LogisticRegression
