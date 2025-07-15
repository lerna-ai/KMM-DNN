/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.multitasknetwork

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import java.io.Serializable

/**
 * The configuration of an output network of a [MultiTaskNetwork].
 *
 * @property outputSize the size of the output layer
 * @property outputActivation the activation function of the output layer
 */
data class MultiTaskNetworkConfig(
  val outputSize: Int,
  val outputActivation: ActivationFunction?
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }
}
