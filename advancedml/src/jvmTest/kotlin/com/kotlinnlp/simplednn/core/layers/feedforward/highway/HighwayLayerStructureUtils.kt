/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward.highway

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.highway.HighwayLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.feedforward.highway.HighwayLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object HighwayLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(): HighwayLayer<DenseNDArray> = HighwayLayer(
    inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.8f, -0.9f, -0.9f, 1.0f))),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray.zeros(4),
    params = buildParams(),
    activationFunction = Tanh,
    dropout = 0.0f
  )

  /**
   *
   */
  fun buildParams() = HighwayLayerParameters(inputSize = 4).apply {

    input.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.5f, 0.6f, -0.8f, -0.6f),
        floatArrayOf(0.7f, -0.4f, 0.1f, -0.8f),
        floatArrayOf(0.7f, -0.7f, 0.3f, 0.5f),
        floatArrayOf(0.8f, -0.9f, 0.0f, -0.1f)
      )))

    transformGate.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.1f, 0.4f, -1.0f, 0.4f),
        floatArrayOf(0.7f, -0.2f, 0.1f, 0.0f),
        floatArrayOf(0.7f, 0.8f, -0.5f, -0.3f),
        floatArrayOf(-0.9f, 0.9f, -0.3f, -0.3f)
      )))

    input.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.0f, -0.3f, 0.8f)))
    transformGate.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.9f, 0.2f, -0.9f, 0.2f)))
  }

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(floatArrayOf(0.57f, 0.75f, -0.15f, 1.64f))
}
