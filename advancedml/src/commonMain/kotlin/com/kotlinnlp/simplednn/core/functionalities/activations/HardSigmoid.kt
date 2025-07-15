/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

/**
 * The Hardsigmoid(x) activation function,
 * transforming the values x into the range [0, 1].
 */

object HardSigmoid: ScalarActivationFunction {

  /**
   * Private val used to serialize the class (needed by Serializable).
   */
  @Suppress("unused")
  private const val serialVersionUID: Long = 1L

  /**
   * Calculate the activation function in [x].
   *
   * @param x input
   *
   * @return f([x])
   */
  override fun f(x: Float): Float = if (x > 2.5) 1.0f else if (x < -2.5) 0.0f else 0.2f * x + 0.5f

  /**
   * Optimized derivative of the HardSigmoid function, calculated respect to the input already activated.
   *
   * @param fx the input already activated [f(x)]
   *
   * @return the HardSigmoid derivative calculated in x
   */
  override fun dfOptimized(fx: Float): Float = if ((fx - 0.5f) / 0.2f < 2.5 && (fx - 0.5f) / 0.2f > -2.5) 0.2f else 0.0f
}
