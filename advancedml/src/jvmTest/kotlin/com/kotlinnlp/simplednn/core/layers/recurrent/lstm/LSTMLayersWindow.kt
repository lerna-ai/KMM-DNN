/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.lstm

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm.LSTMLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.lstm.LSTMLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 *
 */
internal sealed class LSTMLayersWindow: LayersWindow {

  /**
   *
   */
  object Empty : LSTMLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Back : LSTMLayersWindow() {

    override fun getPrevState(): LSTMLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  class BackHidden: LSTMLayersWindow() {

    private lateinit var initHidden: LSTMLayer<DenseNDArray>

    fun setRefLayer(refLayer: LSTMLayer<DenseNDArray>) {
      this.initHidden = buildInitHiddenLayer(refLayer)
    }

    override fun getPrevState(): LSTMLayer<DenseNDArray> = this.initHidden

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  class Front(private val refLayer: LSTMLayer<DenseNDArray>? = null): LSTMLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): LSTMLayer<DenseNDArray> = this.refLayer ?: buildNextStateLayer()
  }

  /**
   *
   */
  object Bilateral : LSTMLayersWindow() {

    override fun getPrevState(): LSTMLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): LSTMLayer<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): LSTMLayer<DenseNDArray>  = LSTMLayer(
  inputArray = AugmentedArray<DenseNDArray>(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2f, 0.2f, -0.3f, -0.9f, -0.8f))).apply {
    activate()
  },
  params = LSTMLayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = LSTMLayersWindow.Empty,
  dropout = 0.0f
).apply {
  cell.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.8f, -0.6f, 1.0f, 0.1f, 0.1f)))
  cell.activate()
}

/**
 *
 */
private fun buildInitHiddenLayer(refLayer: LSTMLayer<DenseNDArray>): LSTMLayer<DenseNDArray> = LSTMLayer(
  inputArray = AugmentedArray(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2f, 0.2f, -0.3f, -0.9f, -0.8f))),
  params = refLayer.params,
  activationFunction = Tanh,
  layersWindow = LSTMLayersWindow.Front(refLayer),
  dropout = 0.0f
)

/**
 *
 */
private fun buildNextStateLayer(): LSTMLayer<DenseNDArray> = LSTMLayer(
  inputArray = AugmentedArray<DenseNDArray>(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(5))).apply {
    assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.1f, -0.5f, 0.7f, 0.2f)))
  },
  params = LSTMLayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = LSTMLayersWindow.Empty,
  dropout = 0.0f
).apply {
  inputGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, -0.3f, -0.2f, 0.3f, 0.6f)))
  outputGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.9f, 0.2f, -0.5f, 1.0f)))
  forgetGate.assignValues(values = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.3f, -0.4f, 0.9f, -0.8f, -0.4f)))
  forgetGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.4f, 0.6f, -0.1f, 0.3f, 0.0f)))
  candidate.assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.4f, 0.2f, -1.0f, 0.7f, -0.3f)))
  cell.assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.3f, 0.8f, 1.0f, -0.4f, 0.6f)))
}
