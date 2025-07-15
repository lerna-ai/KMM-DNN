/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.recurrent.ltm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm.LTMLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.ltm.LTMLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal sealed class LTMLayersWindow: LayersWindow {

  /**
   *
   */
  object Empty : LTMLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Back : LTMLayersWindow() {

    override fun getPrevState(): LTMLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  class Front(private val refLayer: LTMLayer<DenseNDArray>? = null): LTMLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): LTMLayer<DenseNDArray> = this.refLayer ?: buildNextStateLayer()
  }

  /**
   *
   */
  object Bilateral : LTMLayersWindow() {

    override fun getPrevState(): LTMLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): LTMLayer<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): LTMLayer<DenseNDArray> = LTMLayer(
  inputArray = AugmentedArray<DenseNDArray>(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.2, 0.2, -0.3, -0.9))).apply {
    activate()
  },
  params = LTMLayerParameters(inputSize = 4),
  layersWindow = LTMLayersWindow.Empty,
  dropout = 0.0
).apply {
  cell.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.8, -0.6, 1.0, 0.1)))
  cell.activate()
}

/**
 *
 */
private fun buildNextStateLayer(): LTMLayer<DenseNDArray> = LTMLayer(
  inputArray = AugmentedArray.zeros(size = 4),
  inputType = LayerType.Input.Dense,
  outputArray =
  AugmentedArray.zeros(4).apply {
    assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, 0.1, -0.5, 0.7)))
  },
  params = LTMLayerParameters(inputSize = 4),
  layersWindow = LTMLayersWindow.Empty,
  dropout = 0.0
).apply {
  inputArray.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.7, -0.3, -0.2, 0.3)))
  c.assignErrors(errors = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.9, 0.2, -0.5)))
}
