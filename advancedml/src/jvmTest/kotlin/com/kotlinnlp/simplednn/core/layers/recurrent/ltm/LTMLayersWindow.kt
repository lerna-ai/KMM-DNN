/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.ltm

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
  outputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2f, 0.2f, -0.3f, -0.9f))).apply {
    activate()
  },
  params = LTMLayerParameters(inputSize = 4),
  layersWindow = LTMLayersWindow.Empty,
  dropout = 0.0f
).apply {
  cell.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.8f, -0.6f, 1.0f, 0.1f)))
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
    assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.1f, -0.5f, 0.7f)))
  },
  params = LTMLayerParameters(inputSize = 4),
  layersWindow = LTMLayersWindow.Empty,
  dropout = 0.0f
).apply {
  inputArray.assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, -0.3f, -0.2f, 0.3f)))
  c.assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.9f, 0.2f, -0.5f)))
}
