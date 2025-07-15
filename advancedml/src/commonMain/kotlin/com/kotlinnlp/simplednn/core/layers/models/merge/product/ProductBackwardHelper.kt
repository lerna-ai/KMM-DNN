/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.product

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [ProductLayer].
 *
 * @property layer the layer in which the backward is executed
 */
internal class ProductBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: ProductLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * Assign the the layer gradients.
   *
   * gxi = gy * prod(xj) [j != i]
   */
  private fun assignLayerGradients() {

    val gy: DenseNDArray = this.layer.outputArray.errors

    this.layer.inputArrays.forEachIndexed { i, xi ->

      val j0: Int = if (i == 0) 1 else 0
      val prod: DenseNDArray = gy.prod(this.layer.inputArrays[j0].values)

      this.layer.inputArrays.forEachIndexed { j, xj ->
        if (j != j0 && j != i) prod.assignProd(xj.values)
      }

      xi.assignErrors(prod)
    }
  }
}
