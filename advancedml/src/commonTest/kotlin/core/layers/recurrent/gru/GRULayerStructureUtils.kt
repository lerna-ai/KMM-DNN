/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.gru

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
    inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9, -0.9, 1.0))),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(5))),
    params = buildParams(),
    activationFunction = Tanh,
    layersWindow = layersWindow,
    dropout = 0.0
  )

  /**
   *
   */
  fun buildParams(): GRULayerParameters = GRULayerParameters(inputSize = 4, outputSize = 5).apply {

    resetGate.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.5, 0.6, -0.8, -0.6),
        doubleArrayOf(0.7, -0.4, 0.1, -0.8),
        doubleArrayOf(0.7, -0.7, 0.3, 0.5),
        doubleArrayOf(0.8, -0.9, 0.0, -0.1),
        doubleArrayOf(0.4, 1.0, -0.7, 0.8)
      )))

    partitionGate.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.1, 0.4, -1.0, 0.4),
        doubleArrayOf(0.7, -0.2, 0.1, 0.0),
        doubleArrayOf(0.7, 0.8, -0.5, -0.3),
        doubleArrayOf(-0.9, 0.9, -0.3, -0.3),
        doubleArrayOf(-0.7, 0.6, -0.6, -0.8)
      )))

    candidate.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(-1.0, 0.2, 0.0, 0.2),
        doubleArrayOf(-0.7, 0.7, -0.3, -0.3),
        doubleArrayOf(0.3, -0.6, 0.0, 0.7),
        doubleArrayOf(-1.0, -0.6, 0.9, 0.8),
        doubleArrayOf(0.5, 0.8, -0.9, -0.8)
      )))

    resetGate.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.4, 0.0, -0.3, 0.8, -0.4)))
    partitionGate.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.9, 0.2, -0.9, 0.2, -0.9)))
    candidate.biases.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, -0.5, 1.0, 0.4, 0.9)))

    resetGate.recurrentWeights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.0, 0.8, 0.8, -1.0, -0.7),
        doubleArrayOf(-0.7, -0.8, 0.2, -0.7, 0.7),
        doubleArrayOf(-0.9, 0.9, 0.7, -0.5, 0.5),
        doubleArrayOf(0.0, -0.1, 0.5, -0.2, -0.8),
        doubleArrayOf(-0.6, 0.6, 0.8, -0.1, -0.3)
      )))

    partitionGate.recurrentWeights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.1, -0.6, -1.0, -0.1, -0.4),
        doubleArrayOf(0.5, -0.9, 0.0, 0.8, 0.3),
        doubleArrayOf(-0.3, -0.9, 0.3, 1.0, -0.2),
        doubleArrayOf(0.7, 0.2, 0.3, -0.4, -0.6),
        doubleArrayOf(-0.2, 0.5, -0.2, -0.9, 0.4)
      )))

    candidate.recurrentWeights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.2, -0.3, -0.3, -0.5, -0.7),
        doubleArrayOf(0.4, -0.1, -0.6, -0.4, -0.8),
        doubleArrayOf(0.6, 0.6, 0.1, 0.7, -0.4),
        doubleArrayOf(-0.8, 0.9, 0.1, -0.1, -0.2),
        doubleArrayOf(-0.5, -0.3, -0.6, -0.6, 0.1)
      )))
  }

  /**
   *
   */
  fun getOutputGold() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.57, 0.75, -0.15, 1.64, 0.45))
}
