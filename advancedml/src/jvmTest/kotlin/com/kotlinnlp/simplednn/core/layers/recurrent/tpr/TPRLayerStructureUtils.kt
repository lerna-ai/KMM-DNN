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
internal object TPRLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(layersWindow: LayersWindow): TPRLayer<DenseNDArray> = TPRLayer(
    inputArray = AugmentedArray(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.8f, -0.9f, 0.9f, 0.1f))),
    inputType = LayerType.Input.Dense,
    params = buildParams(),
    layersWindow = layersWindow,
    q = 0.001f,
    dropout = 0.0f
  )

  /**
   *
   */
  fun buildParams() = TPRLayerParameters(inputSize = 4, dRoles = 2, dSymbols = 3, nRoles = 3, nSymbols = 4).apply {

    wInS.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.2f, 0.1f, 0.3f, -0.4f),
        floatArrayOf(0.3f, -0.1f, 0.9f, 0.3f),
        floatArrayOf(0.4f, 0.2f, -0.3f, 0.1f),
        floatArrayOf(0.6f, 0.5f, -0.4f, 0.5f)
      )))

    wInR.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.3f, 0.5f, -0.5f, -0.5f),
        floatArrayOf(0.5f, 0.4f, 0.1f, 0.3f),
        floatArrayOf(0.6f, 0.7f, 0.8f, 0.6f)
      )))

    wRecS.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.4f, 0.2f, -0.4f, 0.5f, 0.2f, -0.5f),
        floatArrayOf(-0.2f, 0.7f, 0.8f, -0.5f, 0.5f, 0.7f),
        floatArrayOf(0.4f, -0.1f, 0.1f, 0.7f, -0.1f, 0.3f),
        floatArrayOf(0.3f, 0.2f, -0.7f, -0.8f, -0.3f, 0.6f)
      )))

    wRecR.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.4f, 0.8f, -0.4f, 0.7f, 0.2f, -0.5f),
        floatArrayOf(-0.2f, 0.7f, 0.8f, -0.5f, 0.3f, 0.7f),
        floatArrayOf(0.3f, -0.1f, 0.1f, 0.3f, -0.1f, 0.2f)
      )))

    bS.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, 0.4f, 0.8f, 0.6f)))
    bR.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, 0.2f, -0.1f)))

    s.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.3f, -0.2f, -0.1f, 0.5f),
        floatArrayOf(0.6f, 0.7f, 0.5f, -0.6f),
        floatArrayOf(0.4f, 0.2f, 0.5f, -0.6f)
      )))

    r.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.4f, 0.3f, 0.3f),
        floatArrayOf(0.3f, 0.2f, 0.1f)
      )))
  }

  /**
   *
   */
  fun getOutputGold() = DenseNDArrayFactory.arrayOf(floatArrayOf(0.57f, 0.75f, -0.15f, 1.64f, 0.45f, 0.11f))
}
