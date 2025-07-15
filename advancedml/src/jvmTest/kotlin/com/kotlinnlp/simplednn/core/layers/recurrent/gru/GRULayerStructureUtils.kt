/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.gru

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.gru.GRULayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.gru.GRULayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape


/**
 *
 */
internal object GRULayerStructureUtils {

  /**
   *
   */
  fun buildLayer(layersWindow: LayersWindow): GRULayer<DenseNDArray> = GRULayer(
    inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.8f, -0.9f, -0.9f, 1.0f))),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(5))),
    params = buildParams(),
    activationFunction = Tanh,
    layersWindow = layersWindow,
    dropout = 0.0f
  )

  /**
   *
   */
  fun buildParams(): GRULayerParameters = GRULayerParameters(inputSize = 4, outputSize = 5).apply {

    resetGate.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.5f, 0.6f, -0.8f, -0.6f),
        floatArrayOf(0.7f, -0.4f, 0.1f, -0.8f),
        floatArrayOf(0.7f, -0.7f, 0.3f, 0.5f),
        floatArrayOf(0.8f, -0.9f, 0.0f, -0.1f),
        floatArrayOf(0.4f, 1.0f, -0.7f, 0.8f)
      )))

    partitionGate.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.1f, 0.4f, -1.0f, 0.4f),
        floatArrayOf(0.7f, -0.2f, 0.1f, 0.0f),
        floatArrayOf(0.7f, 0.8f, -0.5f, -0.3f),
        floatArrayOf(-0.9f, 0.9f, -0.3f, -0.3f),
        floatArrayOf(-0.7f, 0.6f, -0.6f, -0.8f)
      )))

    candidate.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(-1.0f, 0.2f, 0.0f, 0.2f),
        floatArrayOf(-0.7f, 0.7f, -0.3f, -0.3f),
        floatArrayOf(0.3f, -0.6f, 0.0f, 0.7f),
        floatArrayOf(-1.0f, -0.6f, 0.9f, 0.8f),
        floatArrayOf(0.5f, 0.8f, -0.9f, -0.8f)
      )))

    resetGate.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.0f, -0.3f, 0.8f, -0.4f)))
    partitionGate.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.9f, 0.2f, -0.9f, 0.2f, -0.9f)))
    candidate.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, -0.5f, 1.0f, 0.4f, 0.9f)))

    resetGate.recurrentWeights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.0f, 0.8f, 0.8f, -1.0f, -0.7f),
        floatArrayOf(-0.7f, -0.8f, 0.2f, -0.7f, 0.7f),
        floatArrayOf(-0.9f, 0.9f, 0.7f, -0.5f, 0.5f),
        floatArrayOf(0.0f, -0.1f, 0.5f, -0.2f, -0.8f),
        floatArrayOf(-0.6f, 0.6f, 0.8f, -0.1f, -0.3f)
      )))

    partitionGate.recurrentWeights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.1f, -0.6f, -1.0f, -0.1f, -0.4f),
        floatArrayOf(0.5f, -0.9f, 0.0f, 0.8f, 0.3f),
        floatArrayOf(-0.3f, -0.9f, 0.3f, 1.0f, -0.2f),
        floatArrayOf(0.7f, 0.2f, 0.3f, -0.4f, -0.6f),
        floatArrayOf(-0.2f, 0.5f, -0.2f, -0.9f, 0.4f)
      )))

    candidate.recurrentWeights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.2f, -0.3f, -0.3f, -0.5f, -0.7f),
        floatArrayOf(0.4f, -0.1f, -0.6f, -0.4f, -0.8f),
        floatArrayOf(0.6f, 0.6f, 0.1f, 0.7f, -0.4f),
        floatArrayOf(-0.8f, 0.9f, 0.1f, -0.1f, -0.2f),
        floatArrayOf(-0.5f, -0.3f, -0.6f, -0.6f, 0.1f)
      )))
  }

  /**
   *
   */
  fun getOutputGold() = DenseNDArrayFactory.arrayOf(floatArrayOf(0.57f, 0.75f, -0.15f, 1.64f, 0.45f))
}
