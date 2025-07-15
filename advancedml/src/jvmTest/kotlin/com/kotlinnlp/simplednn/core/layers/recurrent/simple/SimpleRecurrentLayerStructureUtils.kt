/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.RecurrentLayerUnit
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.simple.SimpleRecurrentLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object SimpleRecurrentLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(layersWindow: LayersWindow) = SimpleRecurrentLayer(
    inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.8f, -0.9f, -0.9f, 1.0f))),
    inputType = LayerType.Input.Dense,
    outputArray = RecurrentLayerUnit(5),
    params = buildParams(),
    activationFunction = Tanh,
    layersWindow = layersWindow,
    dropout = 0.0f
  )

  /**
   *
   */
  fun buildParams() = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5).apply {

    unit.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.5f, 0.6f, -0.8f, -0.6f),
        floatArrayOf(0.7f, -0.4f, 0.1f, -0.8f),
        floatArrayOf(0.7f, -0.7f, 0.3f, 0.5f),
        floatArrayOf(0.8f, -0.9f, 0.0f, -0.1f),
        floatArrayOf(0.4f, 1.0f, -0.7f, 0.8f)
      )))

    unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.0f, -0.3f, 0.8f, -0.4f)))

    unit.recurrentWeights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.0f, 0.8f, 0.8f, -1.0f, -0.7f),
        floatArrayOf(-0.7f, -0.8f, 0.2f, -0.7f, 0.7f),
        floatArrayOf(-0.9f, 0.9f, 0.7f, -0.5f, 0.5f),
        floatArrayOf(0.0f, -0.1f, 0.5f, -0.2f, -0.8f),
        floatArrayOf(-0.6f, 0.6f, 0.8f, -0.1f, -0.3f)
      )))
  }

  /**
   *
   */
  fun getOutputGold() = DenseNDArrayFactory.arrayOf(floatArrayOf(0.57f, 0.75f, -0.15f, 1.64f, 0.45f))
}
