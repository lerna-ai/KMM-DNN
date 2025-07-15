/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * ActivationsFunction can either be used through an [com.kotlinnlp.simplednn.core.arrays.ActivableArray],
 * or through the activation of a [com.kotlinnlp.simplednn.core.layers.Layer]
 */
interface ScalarActivationFunction : ActivationFunction {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * Calculate the activation function in [x].
   *
   * @param x input
   *
   * @return f([x])
   */
  fun f(x: Double): Double

  /**
   * Optimized derivative of the activation function, calculated in [fx].
   *
   * @param fx the input already activated [f(x)]
   *
   * @return the derivative of f calculated in x
   */
  fun dfOptimized(fx: Double): Double

  /**
   * Derivative of the activation function, calculated in [x].
   *
   * @param x the input
   *
   * @return the derivative of f calculated in x
   */
  fun df(x: Double): Double = this.dfOptimized(this.f(x))

  /**
   * Assign to [out] the result of the activation function applied to [array].
   *
   * @param array the input array
   * @param out the NDArray in which the result is written
   */
  override fun f(array: DenseNDArray, out: DenseNDArray) {
    (0 until array.length).forEach { i -> out[i] = this.f(array[i]) }
  }

  /**
   * Assign to [out] the activation function derivative calculated respect to the input array already activated, as
   * optimization.
   *
   * @param fxArray the input array already activated
   * @param out the NDArray in which the result is written
   */
  override fun dfOptimized(fxArray: DenseNDArray, out: DenseNDArray) {
    (0 until fxArray.length).forEach { i -> out[i] = this.dfOptimized(fxArray[i]) }
  }

  /**
   * Assign to [out] the activation function derivative calculated respect to the input array.
   *
   * @param xArray the input array
   * @param out the NDArray in which the result is written
   */
  override fun df(xArray: DenseNDArray, out: DenseNDArray) {
    (0 until xArray.length).forEach { i -> out[i] = this.df(xArray[i]) }
  }
}
