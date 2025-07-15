/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn

import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNParameters
import java.io.Serializable

/**
 * The DeepBiRNNParameters contains the parameters of all its stacked BiRNN.
 *
 * @property paramsPerBiRNN an array of [BiRNNParameters]
 */
class DeepBiRNNParameters(val paramsPerBiRNN: List<BiRNNParameters>) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }
}
