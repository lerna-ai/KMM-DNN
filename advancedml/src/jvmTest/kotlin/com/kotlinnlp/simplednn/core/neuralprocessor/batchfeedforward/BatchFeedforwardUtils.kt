/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor.batchfeedforward

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object BatchFeedforwardUtils {

  /**
   *
   */
  fun buildInputBatch(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.3f, -0.8f)),
    DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, -0.9f, 0.6f)),
    DenseNDArrayFactory.arrayOf(floatArrayOf(0.8f, 0.3f, -0.6f))
  )

  /**
   *
   */
  fun buildOutputErrors(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, -0.2f)),
    DenseNDArrayFactory.arrayOf(floatArrayOf(-0.7f, 0.0f)),
    DenseNDArrayFactory.arrayOf(floatArrayOf(-0.4f, -0.9f))
  )

  /**
   *
   */
  fun buildParams() = StackedLayersParameters(
    LayerInterface(size = 3, type = LayerType.Input.Dense),
    LayerInterface(size = 2, activationFunction = Tanh, connectionType = LayerType.Connection.Feedforward)
  ).apply {

    getLayerParams<FeedforwardLayerParameters>(0).apply {

      unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(-0.7f, 0.3f, -1.0f),
        floatArrayOf(0.8f, -0.6f, 0.4f)
      )))

      unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.2f, -0.9f)))
    }
  }
}
