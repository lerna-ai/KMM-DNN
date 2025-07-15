/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.deltarnn

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.RecurrentLayerUnit
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.deltarnn.DeltaRNNLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.deltarnn.DeltaRNNLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal sealed class DeltaLayersWindow: LayersWindow {

  /**
   *
   */
  object Empty : DeltaLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Back : DeltaLayersWindow() {

    override fun getPrevState(): DeltaRNNLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Front : DeltaLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): DeltaRNNLayer<DenseNDArray> = buildNextStateLayer()
  }

  /**
   *
   */
  object Bilateral : DeltaLayersWindow() {

    override fun getPrevState(): DeltaRNNLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): DeltaRNNLayer<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): DeltaRNNLayer<DenseNDArray> = DeltaRNNLayer(
  inputArray = AugmentedArray(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = RecurrentLayerUnit<DenseNDArray>(5).apply {
    assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2f, 0.2f, -0.3f, -0.9f, -0.8f)))
    setActivation(Tanh)
    activate()
  },
  params = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = DeltaLayersWindow.Empty,
  dropout = 0.0f
)

/**
 *
 */
private fun buildNextStateLayer(): DeltaRNNLayer<DenseNDArray> = DeltaRNNLayer(
  inputArray = AugmentedArray<DenseNDArray>(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = RecurrentLayerUnit<DenseNDArray>(5).apply {
    assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.1f, -0.5f, 0.7f, 0.2f)))
  },
  params = DeltaRNNLayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = DeltaLayersWindow.Empty,
  dropout = 0.0f
).apply {
  wx.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, -0.7f, -0.2f, 0.8f, -0.6f)))
  partition.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2f, -0.1f, 0.6f, -0.8f, 0.5f)))
  candidate.assignErrors(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.4f, 0.6f, -0.1f, 0.3f, 0.0f)))
}
