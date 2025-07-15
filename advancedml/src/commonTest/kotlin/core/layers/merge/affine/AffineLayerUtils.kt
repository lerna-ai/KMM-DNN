/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.affine

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
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(-0.8, -0.9))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, -0.2, 0.6)))),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray(size = 2),
    params = buildParams(),
    activationFunction = Tanh,
    dropout = 0.0
  )

  /**
   *
   */
  fun buildParams() = AffineLayerParameters(inputsSize = listOf(2, 3), outputSize = 2).apply {

    w[0].values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.3, 0.8),
        doubleArrayOf(0.8, -0.7)
      )))

    w[1].values.assignValues(
      DenseNDArrayFactory.arrayOf(listOf(
        doubleArrayOf(0.6, 0.5, -0.9),
        doubleArrayOf(0.3, -0.3, 0.3)
      )))

    b.values.assignValues(DenseNDArrayFactory.arrayOf(doubleArrayOf(0.5, -0.4)))
  }

  /**
   *
   */
  fun getOutputGold() = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.1, -0.3))
}
