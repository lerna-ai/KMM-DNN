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
internal sealed class GRULayersWindow: LayersWindow {

  /**
   *
   */
  object Empty : GRULayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Back : GRULayersWindow() {

    override fun getPrevState(): GRULayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Front : GRULayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): GRULayer<DenseNDArray> = buildNextStateLayer()
  }

  /**
   *
   */
  object Bilateral : GRULayersWindow() {

    override fun getPrevState(): GRULayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): GRULayer<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): GRULayer<DenseNDArray> = GRULayer(
  inputArray = AugmentedArray(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2f, 0.2f, -0.3f, -0.9f, -0.8f))).apply {
    activate()
  },
  params = GRULayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = GRULayersWindow.Empty,
  dropout = 0.0f
)

/**
 *
 */
private fun buildNextStateLayer(): GRULayer<DenseNDArray> = GRULayer(
  inputArray = AugmentedArray<DenseNDArray>(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = AugmentedArray(values = DenseNDArrayFactory.emptyArray(Shape(5))).apply {
    assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.1f, -0.5f, 0.7f, 0.2f)))
  },
  params = GRULayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = GRULayersWindow.Empty,
  dropout = 0.0f
).apply {
  resetGate.assignValues(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.8f, 1.0f, -0.8f, 0.0f, 0.1f)))
  resetGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, -0.3f, -0.2f, 0.3f, 0.6f)))
  partitionGate.assignValues(values = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2f, -0.1f, 0.6f, -0.8f, 0.5f)))
  partitionGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.9f, 0.2f, -0.5f, 1.0f)))
  candidate.assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.4f, 0.6f, -0.1f, 0.3f, 0.0f)))
}
