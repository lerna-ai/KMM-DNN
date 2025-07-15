/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.simple

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.RecurrentLayerUnit
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.recurrent.simple.SimpleRecurrentLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal sealed class SimpleRecurrentLayersWindow: LayersWindow {

  /**
   *
   */
  object Empty : SimpleRecurrentLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Back : SimpleRecurrentLayersWindow() {

    override fun getPrevState(): SimpleRecurrentLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Front : SimpleRecurrentLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): SimpleRecurrentLayer<DenseNDArray> = buildNextStateLayer()
  }

  /**
   *
   */
  object Bilateral : SimpleRecurrentLayersWindow() {

    override fun getPrevState(): SimpleRecurrentLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): SimpleRecurrentLayer<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): SimpleRecurrentLayer<DenseNDArray> = SimpleRecurrentLayer(
  inputArray = AugmentedArray(size = 4),
  outputArray = RecurrentLayerUnit<DenseNDArray>(5).apply {
    assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9, -0.8)))
    setActivation(Tanh)
    activate()
  },
  inputType = LayerType.Input.Dense,
  params = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = SimpleRecurrentLayersWindow.Empty,
  dropout = 0.0
)

/**
 *
 */
private fun buildNextStateLayer(): SimpleRecurrentLayer<DenseNDArray> = SimpleRecurrentLayer(
  inputArray = AugmentedArray(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = RecurrentLayerUnit<DenseNDArray>(5).apply {
    assignErrors(errors =  DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, -0.5, 0.7, 0.2)))
  },
  params = SimpleRecurrentLayerParameters(inputSize = 4, outputSize = 5),
  activationFunction = Tanh,
  layersWindow = SimpleRecurrentLayersWindow.Empty,
  dropout = 0.0
)
