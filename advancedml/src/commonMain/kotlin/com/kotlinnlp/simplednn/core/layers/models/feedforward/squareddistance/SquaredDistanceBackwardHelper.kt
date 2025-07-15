/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.squareddistance

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [SquaredDistanceLayer].
 *
 * @property layer the layer in which the forward is executed
 */
internal class SquaredDistanceBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: SquaredDistanceLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    val gy: DenseNDArray = this.layer.outputArray.errors

    this.layer.bhOut.assignErrors(this.layer.bhOut.valuesNotActivated.assignProd(gy[0] * 2.0))

    this.assignParamsGradients()

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * Assign the the parameters (the B matrix) gradients.
   */
  private fun assignParamsGradients() {

    val gB = this.layer.params.wB.errors.values

    gB.assignDot(this.layer.bhOut.errors, this.layer.inputArray.values.t)
  }

  /**
   * Assign the the layer gradients.
   */
  private fun assignLayerGradients() {

    val b = this.layer.params.wB.values

    this.layer.inputArray.assignErrors(this.layer.bhOut.errors.t.dot(b))
  }
}
