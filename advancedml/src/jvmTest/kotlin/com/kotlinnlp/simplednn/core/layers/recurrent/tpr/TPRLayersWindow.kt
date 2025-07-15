/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.tpr

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.LayersWindow
import com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr.TPRLayer
import com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr.TPRLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal sealed class TPRLayersWindow: LayersWindow {

  /**
   *
   */
  object Empty : TPRLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  object Back : TPRLayersWindow() {

    override fun getPrevState(): TPRLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): Nothing? = null
  }

  /**
   *
   */
  class Front(private val refLayer: TPRLayer<DenseNDArray>? = null): TPRLayersWindow() {

    override fun getPrevState(): Nothing? = null

    override fun getNextState(): TPRLayer<DenseNDArray> = this.refLayer ?: buildNextStateLayer()
  }

  /**
   *
   */
  object Bilateral : TPRLayersWindow() {

    override fun getPrevState(): TPRLayer<DenseNDArray> = buildPrevStateLayer()

    override fun getNextState(): TPRLayer<DenseNDArray> = buildNextStateLayer()
  }
}

/**
 *
 */
private fun buildPrevStateLayer(): TPRLayer<DenseNDArray> = TPRLayer(
  inputArray = AugmentedArray<DenseNDArray>(size = 4),
  inputType = LayerType.Input.Dense,
  params = TPRLayerParameters(inputSize = 4, dRoles = 3, dSymbols = 2, nRoles = 3, nSymbols = 4),
  layersWindow = TPRLayersWindow.Empty,
  q = 0.001f,
  dropout = 0.0f
).apply {
  outputArray.values.assignValues(
    DenseNDArrayFactory.arrayOf(floatArrayOf(0.211f, -0.451f, 0.499f, -1.333f, -0.11645f, 0.366f)))
}

/**
 *
 */
private fun buildNextStateLayer(): TPRLayer<DenseNDArray> = TPRLayer(
  inputArray = AugmentedArray<DenseNDArray>(size = 4),
  inputType = LayerType.Input.Dense,
  params = TPRLayerParameters(inputSize = 4, dRoles = 3, dSymbols = 2, nRoles = 3, nSymbols = 4),
  layersWindow = TPRLayersWindow.Empty,
  q = 0.001f,
  dropout = 0.0f
).apply {

  outputArray.assignErrors(DenseNDArrayFactory.arrayOf(floatArrayOf(0.711f, -0.099f, 0.459f, -1.235f, -0.9845f, 0.9292f)))

  aS.assignValues(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.8f, 1.0f, -0.8f, 0.0f)))
  aS.assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, 0.3f, 0.2f, 0.3f)))
  aR.assignValues(values = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2f, -0.1f, 0.6f)))
  aR.assignErrors(errors = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.9f, 0.2f)))
}
