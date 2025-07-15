/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.core.layers.models.LinearParams
import kotlin.math.sqrt

/**
 * The parameters of the Scaled-Dot Attention Layer.
 *
 * @property inputSize the size of each element of the input sequence
 * @property attentionSize the size of each element of the attention sequence
 * @property outputSize the size of each element of the output sequence
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: Glorot)
 */
class ScaledDotAttentionLayerParameters(
  inputSize: Int,
  val attentionSize: Int,
  outputSize: Int,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = GlorotInitializer()
) : LayerParameters(
  inputSize = inputSize,
  outputSize = outputSize,
  weightsInitializer = weightsInitializer,
  biasesInitializer = biasesInitializer
) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The multiplying factor of the attention calculation.
   */
  internal val attentionFactor: Double = 1.0 / sqrt(this.attentionSize.toDouble())

  /**
   * The queries trainable parameter.
   */
  val queries = LinearParams(inputSize = inputSize, outputSize = this.attentionSize)

  /**
   * The keys trainable parameter.
   */
  val keys = LinearParams(inputSize = inputSize, outputSize = this.attentionSize)

  /**
   * The values trainable parameter.
   */
  val values = LinearParams(inputSize = inputSize, outputSize = this.outputSize)

  /**
   * The list of weights parameters.
   */
  override val weightsList: List<ParamsArray> = listOf(
    this.queries.weights,
    this.keys.weights,
    this.values.weights
  )

  /**
   * The list of biases parameters.
   */
  override val biasesList: List<ParamsArray> = listOf(
    this.queries.biases,
    this.keys.biases,
    this.values.biases
  )

  /**
   * Initialize all parameters values.
   */
  init {
    this.initialize()
  }
}
