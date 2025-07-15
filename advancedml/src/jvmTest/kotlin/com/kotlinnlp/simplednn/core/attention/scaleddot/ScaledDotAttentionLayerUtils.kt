/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.attention.scaleddot

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot.ScaledDotAttentionLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object ScaledDotAttentionLayerUtils {

  /**
   *
   */
  fun buildAttentionParams() =
    ScaledDotAttentionLayerParameters(inputSize = 4, attentionSize = 2, outputSize = 4).apply {
      queries.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.3f, 0.4f, 0.2f, -0.2f),
        floatArrayOf(0.2f, -0.1f, 0.1f, 0.6f)
      )))
      keys.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(-0.2f, 0.8f, 0.6f, 0.8f),
        floatArrayOf(0.2f, 0.8f, 0.0f, -0.6f)
      )))
      values.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.9f, 0.8f, 0.5f, 0.3f),
        floatArrayOf(-0.6f, -0.2f, 0.4f, 0.4f),
        floatArrayOf(-0.6f, -0.7f, -0.4f, 0.6f),
        floatArrayOf(0.8f, -0.8f, 0.8f, -0.7f)
      )))
      queries.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, 0.8f)))
      keys.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.7f, -0.1f)))
      values.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2f, -0.2f, -0.7f, 0.9f)))
    }

  /**
   *
   */
  fun buildInputSequence(): List<AugmentedArray<DenseNDArray>> = listOf(
    AugmentedArray(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.4f, 0.7f, 0.9f, 0.6f))),
    AugmentedArray(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.5f, 0.7f, -0.7f, 0.8f))),
    AugmentedArray(DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, -0.5f, 0.0f, 0.2f)))
  )

  /**
   *
   */
  fun buildOutputErrors(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, -0.3f, -0.7f, -0.5f)),
    DenseNDArrayFactory.arrayOf(floatArrayOf(-0.8f, -0.5f, -0.5f, 0.1f)),
    DenseNDArrayFactory.arrayOf(floatArrayOf(-0.6f, -0.5f, 0.2f, -0.9f))
  )
}
