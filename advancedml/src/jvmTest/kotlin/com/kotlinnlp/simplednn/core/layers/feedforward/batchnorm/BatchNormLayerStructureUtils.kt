/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward.batchnorm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.batchnorm.BatchNormLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.batchnorm.BatchNormLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object BatchNormLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(): BatchNormLayer<DenseNDArray> = BatchNormLayer(
    inputArrays = listOf(
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.8f, -0.7f, -0.5f))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.4f, -0.6f, -0.2f, -0.9f))),
      AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.4f, 0.2f, 0.8f)))),
    inputType = LayerType.Input.Dense,
    params = buildParams())

  /**
   *
   */
  fun buildParams() = BatchNormLayerParameters(inputSize = 4).apply {
    g.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.0f, -0.3f, 0.8f)))
    b.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.9f, 0.2f, -0.9f, 0.2f)))
  }

  /**
   *
   */
  fun getOutputErrors1() = DenseNDArrayFactory.arrayOf(floatArrayOf(-1.0f, -0.2f, 0.4f, 0.6f))

  /**
   *
   */
  fun getOutputErrors2() = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.3f, 0.1f, 0.7f, 0.9f))

  /**
   *
   */
  fun getOutputErrors3() = DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, -0.4f, 0.7f, -0.8f))
}
