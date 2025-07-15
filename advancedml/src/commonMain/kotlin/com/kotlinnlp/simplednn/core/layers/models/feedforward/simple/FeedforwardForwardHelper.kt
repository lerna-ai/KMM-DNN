/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.simple

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [FeedforwardLayer] in which the forward is executed
 */
internal class FeedforwardForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: FeedforwardLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   * y = f(w (dot) x + b)
   */
  override fun forward() {

    this.layer.outputArray.forward(
      w = this.layer.params.unit.weights.values,
      b = this.layer.params.unit.biases.values,
      x = this.layer.inputArray.values)

    this.layer.outputArray.activate()
  }

  /**
   * Forward the input to the output combining it with the parameters, saving the contributions.
   *
   * y = f(w (dot) x + b)
   *
   * @param contributions the support in which to save the contributions of the input respect to the output
   */
  override fun forward(contributions: LayerParameters) {

    require(this.layer.inputArray.values is DenseNDArray) {
      "Forwarding with contributions requires the input to be dense."
    }

    this.forwardArray(
      contributions = (contributions as FeedforwardLayerParameters).unit.weights.values,
      x = this.layer.inputArray.values as DenseNDArray,
      y = this.layer.outputArray.values,
      w = this.layer.params.unit.weights.values,
      b = this.layer.params.unit.biases.values)

    this.layer.outputArray.activate()
  }
}
