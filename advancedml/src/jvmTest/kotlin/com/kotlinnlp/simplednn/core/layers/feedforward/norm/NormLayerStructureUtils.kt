/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward.norm

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.feedforward.norm.NormLayer
import com.kotlinnlp.simplednn.core.layers.models.feedforward.norm.NormLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object NormLayerStructureUtils {

  /**
   *
   */
  fun buildLayer(): NormLayer<DenseNDArray> = NormLayer(
    inputArray = AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.8f, -0.7f, -0.5f))),
    outputArray = AugmentedArray.zeros(size = 4),
    inputType = LayerType.Input.Dense,
    params = buildParams())

  /**
   *
   */
  fun buildParams() = NormLayerParameters(inputSize = 4).apply {
    g.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.0f, -0.3f, 0.8f)))
    b.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.9f, 0.2f, -0.9f, 0.2f)))
  }

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(floatArrayOf(-1.0f, -0.2f, 0.4f, 0.6f))
}
