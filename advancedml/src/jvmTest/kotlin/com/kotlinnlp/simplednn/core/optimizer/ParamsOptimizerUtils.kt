/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.optimizer

import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object ParamsOptimizerUtils {

  /**
   *
   */
  fun buildParams() = FeedforwardLayerParameters(inputSize = 4, outputSize = 2).also {

    it.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      floatArrayOf(0.3f, 0.4f, 0.2f, -0.2f),
      floatArrayOf(0.2f, -0.1f, 0.1f, 0.6f)
    )))

    it.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(
      floatArrayOf(0.3f, -0.4f)
    ))
  }

  /**
   *
   */
  fun buildWeightsErrorsValues1() = DenseNDArrayFactory.arrayOf(listOf(
    floatArrayOf(0.3f, 0.4f, 0.2f, -0.2f),
    floatArrayOf(0.2f, -0.1f, 0.1f, 0.6f)
  ))

  /**
   *
   */
  fun buildBiasesErrorsValues1() = DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, -0.4f))

  /**
   *
   */
  fun buildWeightsErrorsValues2() = DenseNDArrayFactory.arrayOf(listOf(
    floatArrayOf(0.7f, -0.8f, 0.1f, -0.6f),
    floatArrayOf(0.8f, 0.6f, -0.9f, -0.2f)
  ))

  /**
   *
   */
  fun buildBiasesErrorsValues2() = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.9f, 0.1f))
}
