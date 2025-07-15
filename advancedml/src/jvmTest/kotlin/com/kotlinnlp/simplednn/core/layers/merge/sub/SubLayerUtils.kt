/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.merge.sub

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.merge.sub.SubLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.sub.SubLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object SubLayerUtils {

  /**
   *
   */
  fun buildLayer(): SubLayer<DenseNDArray> = SubLayer(
    inputArray1 = AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.4f, -0.3f, 0.2f))),
    inputArray2 = AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.8f, 0.2f, -0.6f, 0.9f))),
    inputType = LayerType.Input.Dense,
    outputArray = AugmentedArray(size = 4),
    params = SubLayerParameters(inputSize = 4))

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(floatArrayOf(-1.0f, -0.2f, 0.4f, 0.0f))
}
