/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.concat

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [ConcatLayer].
 *
 * @property layer the layer in which the forward is executed
 */
internal class ConcatForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: ConcatLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output concatenating the input arrays.
   * TODO: make it working with all types of input arrays.
   */
  override fun forward() {

    this.layer.outputArray.assignValues(
      concatVectorsV(this.layer.inputArrays.map { it.values as DenseNDArray })
    )
  }
}
