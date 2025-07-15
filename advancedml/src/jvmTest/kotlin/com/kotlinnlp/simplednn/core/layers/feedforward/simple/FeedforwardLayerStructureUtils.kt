/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward.simple

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArrayFactory

/**
 *
 */
internal object FeedforwardLayerStructureUtils {

  /**
   *
   */
  fun buildLayer45(): FeedforwardLayer<DenseNDArray> = FeedforwardLayer(
    inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.8f, -0.9f, -0.9f, 1.0f))),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray.zeros(5),
    params = getParams45(),
    activationFunction = Tanh,
    dropout = 0.0f
  )

  /**
   *
   */
  fun getParams45() = FeedforwardLayerParameters(inputSize = 4, outputSize = 5).apply {

    unit.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.5f, 0.6f, -0.8f, -0.6f),
        floatArrayOf(0.7f, -0.4f, 0.1f, -0.8f),
        floatArrayOf(0.7f, -0.7f, 0.3f, 0.5f),
        floatArrayOf(0.8f, -0.9f, 0.0f, -0.1f),
        floatArrayOf(0.4f, 1.0f, -0.7f, 0.8f)
      )))

    unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.0f, -0.3f, 0.8f, -0.4f)))
  }

  /**
   *
   */
  fun getOutputGold5(): DenseNDArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.5f, -0.4f, -0.9f, 0.9f))

  /**
   *
   */
  fun buildLayer53(): FeedforwardLayer<DenseNDArray> = FeedforwardLayer(
    inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.4f, -0.8f, 0.0f, 0.7f, -0.2f))),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray.zeros(3),
    params = getParams53(),
    activationFunction = Softmax(),
    dropout = 0.0f
  )

  /**
   *
   */
  fun buildLayer53SparseBinary(): FeedforwardLayer<SparseBinaryNDArray> {

    val input: SparseBinaryNDArray = SparseBinaryNDArrayFactory.arrayOf(activeIndices = listOf(2, 4), shape = Shape(5))

    return FeedforwardLayer(
      inputArray = AugmentedArray(input).apply { setActivation(Tanh) },
      inputType = LayerType.Input.SparseBinary,
      outputArray = AugmentedArray.zeros(3),
      params = getParams53(),
      activationFunction = Softmax(),
      dropout = 0.0f)
  }

  /**
   *
   */
  fun getParams53() = FeedforwardLayerParameters(inputSize = 5, outputSize = 3).apply {

    unit.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.8f, -0.8f, 0.9f, -1.0f, -0.1f),
        floatArrayOf(0.9f, 0.6f, 0.7f, 0.6f, 0.6f),
        floatArrayOf(-0.1f, 0.0f, 0.3f, 0.0f, 0.3f)
      )))

    unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.5f, 0.1f, 0.2f)))
  }

  /**
   *
   */
  fun getOutputGold3(): DenseNDArray = DenseNDArrayFactory.arrayOf(floatArrayOf(1.0f, 0.0f, 0.0f))
}
