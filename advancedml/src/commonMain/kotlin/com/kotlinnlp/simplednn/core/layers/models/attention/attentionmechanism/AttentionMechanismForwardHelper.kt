/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism

import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The helper which executes the forward on a [layer].
 *
 * @property layer the [AttentionMechanismLayer] in which the forward is executed
 */
internal class AttentionMechanismForwardHelper(
  override val layer: AttentionMechanismLayer
) : ForwardHelper<DenseNDArray>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   *    am = attention matrix
   *    cv = context vector
   *
   *    y = activation(am (dot) cv)
   */
  override fun forward() {

    this.layer.outputArray.assignValues(this.layer.attentionMatrix.values.dot(this.layer.params.contextVector.values))
    this.layer.outputArray.activate()
  }
}
