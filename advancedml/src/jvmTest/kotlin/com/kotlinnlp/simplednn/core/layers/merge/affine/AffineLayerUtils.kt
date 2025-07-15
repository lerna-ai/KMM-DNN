/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.merge.affine

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.merge.affine.AffineLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.affine.AffineLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object AffineLayerUtils {

  /**
   *
   */
  fun buildLayer(): AffineLayer<DenseNDArray> = AffineLayer(
    inputArrays = listOf(
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.8f, -0.9f))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, -0.2f, 0.6f)))),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray(size = 2),
    params = buildParams(),
    activationFunction = Tanh,
    dropout = 0.0f
  )

  /**
   *
   */
  fun buildParams() = AffineLayerParameters(inputsSize = listOf(2, 3), outputSize = 2).apply {

    w[0].values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.3f, 0.8f),
        floatArrayOf(0.8f, -0.7f)
      )))

    w[1].values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.6f, 0.5f, -0.9f),
        floatArrayOf(0.3f, -0.3f, 0.3f)
      )))

    b.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, -0.4f)))
  }

  /**
   *
   */
  fun getOutputGold() = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, -0.3f))
}
