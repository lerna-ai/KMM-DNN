/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

import kotlin.math.exp

/**
 *
 * The ELU activation function acts like a [ReLU] if x is positive,
 * but for negative values it is a function bounded by a fixed value -1, for alpha = 1.0
 *
 * References
 * [Djork-Arné Clevert, Thomas Unterthiner, Sepp Hochreiter (2015): Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs)](http://arxiv.org/abs/1511.07289)
 *
 * @property alpha defines the decreasing exponential rate for the negative values. It must be positive.
 */
class ELU(val alpha: Float = 1.0f) : ScalarActivationFunction {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Assert requirements.
   */
  init {
    require(this.alpha > 0.0)
  }

  /**
   * Calculate the ELU function in [x].
   * [alternative form: max(0, x) + min(0, alpha * (exp(x) - 1)))]
   *
   * @param x input
   *
   * @return f([x])
   */
  override fun f(x: Float): Float = if (x > 0) x else this.alpha * (exp(x) - 1.0f)

  /**
   * Optimized derivative of the ELU function, calculated respect to the input already activated.
   *
   * @param fx the input already activated [f(x)]
   *
   * @return the ELU derivative calculated in x
   */
  override fun dfOptimized(fx: Float): Float = if (fx > 0.0) 1.0f else fx + this.alpha
}
