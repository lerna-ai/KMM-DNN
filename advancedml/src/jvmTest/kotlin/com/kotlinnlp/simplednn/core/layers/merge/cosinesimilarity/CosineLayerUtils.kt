/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.merge.cosinesimilarity

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.layers.models.merge.cosinesimilarity.CosineLayer
import com.kotlinnlp.simplednn.core.layers.models.merge.cosinesimilarity.CosineLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
internal object CosineLayerUtils {

  /**
   *
   */
  fun buildLayer(): CosineLayer = CosineLayer(
      inputArray1 = AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.7f, -0.7f, 0.8f, 0.6f))),
      inputArray2 = AugmentedArray(values = DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, -0.4f, 0.8f, -0.7f))),
      params = CosineLayerParameters(inputSize = 4))

  /**
   *
   */
  fun getOutputErrors() = DenseNDArrayFactory.arrayOf(floatArrayOf(0.95f))
}
