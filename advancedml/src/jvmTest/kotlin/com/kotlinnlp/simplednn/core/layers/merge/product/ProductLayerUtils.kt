/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.merge.product

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.merge.product.ProductLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.product.ProductLayer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object ProductLayerUtils {

  /**
   *
   */
  fun buildLayer(): ProductLayer<DenseNDArray> = ProductLayer(
    inputArrays = listOf(
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.9f, 0.9f, 0.6f))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.5f, -0.5f))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.7f, -0.7f, 0.8f))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, -0.4f, -0.8f))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, -0.9f, -0.5f)))
    ),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray(size = 3),
    params = ProductLayerParameters(inputSize = 3, nInputs = 5)
  )

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, -0.5f, -0.8f))
}
