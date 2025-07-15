/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.merge.biaffine

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.merge.biaffine.BiaffineLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.biaffine.BiaffineLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object BiaffineLayerUtils {

  /**
   *
   */
  fun buildLayer(): BiaffineLayer<DenseNDArray> = BiaffineLayer(
    inputArray1 = AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.8f, -0.9f))),
    inputArray2 = AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, -0.2f, 0.6f))),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray(size = 2),
    params = buildParams(),
    activationFunction = Tanh,
    dropout = 0.0f
  )

  /**
   *
   */
  fun buildParams() = BiaffineLayerParameters(inputSize1 = 2, inputSize2 = 3, outputSize = 2).apply {

    w1.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.3f, 0.8f),
        floatArrayOf(0.8f, -0.7f)
      )))

    w2.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.6f, 0.5f, -0.9f),
        floatArrayOf(0.3f, -0.3f, 0.3f)
      )))

    b.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, -0.4f)))

    w[0].values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(-0.4f, 0.2f),
        floatArrayOf(0.2f, 0.4f),
        floatArrayOf(0.0f, 0.5f)
      )))

    w[1].values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(-0.2f, 0.9f),
        floatArrayOf(0.5f, 0.0f),
        floatArrayOf(-0.1f, -0.1f)
      )))
  }

  /**
   *
   */
  fun getOutputGold() = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, -0.3f))
}
