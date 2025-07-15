/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.recurrent.cfn

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.Layer
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [CFNLayer] in which the forward is executed
 */
internal class CFNForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: CFNLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * y = inG * c + f(yPrev) * forG
   */
  override fun forward() {

    val prevStateLayer = this.layer.layersWindow.getPrevState()

    this.setGates(prevStateLayer) // must be called before accessing to the activated values of the gates

    val y: DenseNDArray = this.layer.outputArray.values
    val c: DenseNDArray = this.layer.candidate.values
    val inG: DenseNDArray = this.layer.inputGate.values
    val forG: DenseNDArray = this.layer.forgetGate.values

    // y = inG * c
    y.assignProd(inG, c)

    // y += f(yPrev) * forG
    if (prevStateLayer != null) {
      val yPrev = prevStateLayer.outputArray.values

      this.layer.activatedPrevOutput = if (this.layer.activationFunction != null)
        this.layer.activationFunction.f(yPrev)
      else
        yPrev

      y.assignSum(this.layer.activatedPrevOutput!!.prod(forG))
    }
  }

  /**
   * Set gates values
   *
   * inG = sigmoid(wIn (dot) x + bIn + wrIn (dot) yPrev)
   * forG = sigmoid(wForG (dot) x + bForG + wrForG (dot) yPrev)
   * c = f(wc (dot) x)
   */
  private fun setGates(prevStateLayer: Layer<*>?) {

    val x: InputNDArrayType = this.layer.inputArray.values
    val c: DenseNDArray = this.layer.candidate.values
    val wc: DenseNDArray = this.layer.params.candidateWeights.values

    this.layer.inputGate.forward(
      w = this.layer.params.inputGate.weights.values,
      b = this.layer.params.inputGate.biases.values,
      x = x
    )

    this.layer.forgetGate.forward(
      w = this.layer.params.forgetGate.weights.values,
      b = this.layer.params.forgetGate.biases.values,
      x = x
    )

    c.assignDot(wc, x)

    if (prevStateLayer != null) { // recurrent contribution for input and forget gates
      val yPrev = prevStateLayer.outputArray.valuesNotActivated
      this.layer.inputGate.addRecurrentContribution(this.layer.params.inputGate, yPrev)
      this.layer.forgetGate.addRecurrentContribution(this.layer.params.forgetGate, yPrev)
    }

    this.layer.inputGate.activate()
    this.layer.forgetGate.activate()
    this.layer.candidate.activate()
  }
}
