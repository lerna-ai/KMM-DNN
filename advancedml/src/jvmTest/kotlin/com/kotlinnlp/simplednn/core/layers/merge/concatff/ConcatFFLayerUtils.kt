/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.merge.concatff

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.merge.concatff.ConcatFFLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.concatff.ConcatFFLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object ConcatFFLayerUtils {

  /**
   *
   */
  fun buildLayer(): ConcatFFLayer<DenseNDArray> = ConcatFFLayer(
    inputArrays = listOf(
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.9f, 0.9f, 0.6f, 0.1f))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.5f))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.7f, -0.7f, 0.8f)))
    ),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray.zeros(3),
    params = buildParams(),
    activationFunction = Tanh,
    dropout = 0.0f
  )

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.7f, -0.8f, 0.6f))

  /**
   *
   */
  private fun buildParams() = ConcatFFLayerParameters(inputsSize = listOf(4, 2, 3), outputSize = 3).apply {

    output.unit.weights.values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(-0.1f, -0.3f, 0.5f, 0.6f, -0.6f, 0.6f, 0.4f, -0.2f, -0.9f),
        floatArrayOf(0.6f, 0.6f, -0.2f, 0.3f, 0.7f, -0.2f, 0.9f, -0.3f, -0.5f),
        floatArrayOf(0.7f, 0.7f, 0.0f, -0.1f, -0.9f, 0.4f, 0.2f, 0.1f, -0.4f)
      ))
    )

    output.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, 0.2f, -0.7f)))
  }
}
