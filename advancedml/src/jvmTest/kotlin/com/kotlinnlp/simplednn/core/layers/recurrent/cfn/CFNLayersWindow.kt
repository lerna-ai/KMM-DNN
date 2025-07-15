/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.cfn

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.cfn.CFNLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.cfn.CFNLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape

/**
 *
 */
internal sealed class CFNLayersWindow: LayersWindow {

  /**
   *
   */
  object Empty : CFNLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Back : CFNLayersWindow() {

    override fun getPrevState(): CFNLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  class Front(val currentLayerOutput: DenseNDArray): CFNLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): CFNLayer<DenseNDArray> = buildNextStateLayer(currentLayerOutput)
  }

  /**
   *
   */
  class Bilateral(val currentLayerOutput: DenseNDArray): CFNLayersWindow() {

    override fun getPrevState(): CFNLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): CFNLayer<DenseNDArray> = buildNextStateLayer(currentLayerOutput)
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): CFNLayer<DenseNDArray> = CFNLayer(
  inputArray = AugmentedArray(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2f, 0.2f, -0.3f, -0.9f, -0.8f))).apply {
    activate()
  },
  params = CFNLayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = CFNLayersWindow.Empty,
  dropout = 0.0f
)

/**
 *
 */
private fun buildNextStateLayer(currentLayerOutput: DenseNDArray): CFNLayer<DenseNDArray> = CFNLayer(
  inputArray = AugmentedArray<DenseNDArray>(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = AugmentedArray(DenseNDArrayFactory.emptyArray(Shape(5))).apply {
    assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.1f, -0.5f, 0.7f, 0.2f)))
  },
  params = CFNLayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = CFNLayersWindow.Empty,
  dropout = 0.0f
).apply {
  inputGate.assignValues(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.8f, 1.0f, -0.8f, 0.0f, 0.1f)))
  inputGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, -0.3f, -0.2f, 0.3f, 0.6f)))
  forgetGate.assignValues(values = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2f, -0.1f, 0.6f, -0.8f, 0.5f)))
  forgetGate.assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.9f, 0.2f, -0.5f, 1.0f)))
  activatedPrevOutput = currentLayerOutput
}
