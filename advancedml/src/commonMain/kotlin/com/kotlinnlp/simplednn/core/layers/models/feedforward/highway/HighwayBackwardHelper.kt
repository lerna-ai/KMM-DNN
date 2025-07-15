/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.highway

import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.core.arrays.getInputErrors
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the backward on a [layer].
 *
 * @property layer the [HighwayLayer] in which the backward is executed
 */
internal class HighwayBackwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: HighwayLayer<InputNDArrayType>
) : BackwardHelper<InputNDArrayType>(layer) {

  /**
   * Executes the backward calculating the errors of the parameters and eventually of the input through the SGD
   * algorithm, starting from the preset errors of the output array.
   *
   * @param propagateToInput whether to propagate the errors to the input array
   */
  override fun execBackward(propagateToInput: Boolean) {

    // TODO: extend for all input types
    require(this.layer.inputArray.values is DenseNDArray) { "Highway layer supports only dense input." }

    this.assignGatesGradients()

    this.assignParamsGradients()

    if (propagateToInput) {
      this.assignLayerGradients()
    }
  }

  /**
   * Assign gates gradients.
   */
  private fun assignGatesGradients() {

    val x: InputNDArrayType = this.layer.inputArray.values
    val gy: DenseNDArray = this.layer.outputArray.errors
    val inputUnit: DenseNDArray = this.layer.inputUnit.values
    val tGate: DenseNDArray = this.layer.transformGate.values
    val tDeriv: DenseNDArray = this.layer.transformGate.calculateActivationDeriv()

    this.layer.transformGate.assignErrorsByProd(inputUnit.sub(x as DenseNDArray), gy)
    this.layer.transformGate.errors.assignProd(tDeriv)

    this.layer.inputUnit.assignErrorsByProd(tGate, gy)

    if (this.layer.inputUnit.hasActivation) {
      val inDeriv: DenseNDArray = this.layer.inputUnit.calculateActivationDeriv()
      this.layer.inputUnit.errors.assignProd(inDeriv)
    }
  }

  /**
   * For each gate:
   *   gb = gy * 1
   *   gw = gy (dot) x
   */
  private fun assignParamsGradients() {

    this.layer.inputUnit.assignParamsGradients(
      gw = this.layer.params.input.weights.errors.values,
      gb = this.layer.params.input.biases.errors.values,
      x = this.layer.inputArray.values)

    this.layer.transformGate.assignParamsGradients(
      gw = this.layer.params.transformGate.weights.errors.values,
      gb = this.layer.params.transformGate.biases.errors.values,
      x = this.layer.inputArray.values)
  }

  /**
   * gx = (1 - T) * gy + gxIn + gxT
   */
  private fun assignLayerGradients() {

    val tGate: DenseNDArray = this.layer.transformGate.values
    val gxIn: DenseNDArray = this.layer.inputUnit.getInputErrors(w = this.layer.params.input.weights.values)
    val gxT: DenseNDArray = this.layer.transformGate.getInputErrors(w = this.layer.params.transformGate.weights.values)
    val gy: DenseNDArray = this.layer.outputArray.errors

    this.layer.inputArray.assignErrors(tGate.reverseSub(1.0).assignProd(gy).assignSum(gxIn).assignSum(gxT))
  }
}
