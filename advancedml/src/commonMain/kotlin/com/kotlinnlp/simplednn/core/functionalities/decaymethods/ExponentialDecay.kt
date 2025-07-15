/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.decaymethods

import kotlin.math.exp
import kotlin.math.ln

/**
 * ExponentialDecay defines an exponential decay depending on the time step
 * => LR = exp((iterations - t) * log(LR) + log(LRfinal))
 *
 * @property initLearningRate the initial Learning rate (must be >= [finalLearningRate])
 * @property finalLearningRate the final value which the learning rate will reach (must be >= 0)
 * @property totalIterations total amount of iterations (must be >= 0)
 */
class ExponentialDecay(
  val initLearningRate: Float = 0.0f,
  val finalLearningRate: Float = 0.0f,
  val totalIterations: Int
) : DecayMethod {

  /**
   *
   */
  init { require(this.initLearningRate > this.finalLearningRate) }

  /**
   * Update the learning rate given a time step.
   *
   * @param learningRate the learning rate to decrease
   * @param timeStep the current time step
   *
   * @return the updated learning rate
   */
  override fun update(learningRate: Float, timeStep: Int): Float {
    return if (learningRate > this.finalLearningRate && timeStep > 1) {
      exp(
        ((this.totalIterations - timeStep) * ln(learningRate) + ln(this.finalLearningRate))
          /
          (this.totalIterations - timeStep + 1))
    } else {
      learningRate
    }
  }
}
