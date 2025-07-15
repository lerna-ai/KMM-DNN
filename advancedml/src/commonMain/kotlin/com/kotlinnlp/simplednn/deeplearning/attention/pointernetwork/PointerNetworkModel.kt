/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.pointernetwork

import com.kotlinnlp.simplednn.core.functionalities.initializers.GlorotInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.activations.SoftmaxBase
import com.kotlinnlp.simplednn.core.layers.LayerInterface
import com.kotlinnlp.simplednn.core.layers.StackedLayersParameters
import com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism.AttentionMechanismLayerParameters
import com.kotlinnlp.simplednn.core.layers.models.merge.mergeconfig.*
import java.io.Serializable


/**
 * The model of the [PointerNetworkProcessor].
 *
 * @property inputSize the size of the elements of the input sequence
 * @property vectorSize the size of the vector that modulates a content-based attention mechanism over the input sequence
 * @param activation the activation function of the attention mechanism (default SoftmaxBase)
 * @param mergeConfig the configuration of the merge network
 * @param weightsInitializer the initializer of the weights (zeros if null, default: Glorot)
 * @param biasesInitializer the initializer of the biases (zeros if null, default: null)
 */
class PointerNetworkModel(
  val inputSize: Int,
  val vectorSize: Int,
  internal val activation: ActivationFunction = SoftmaxBase(),
  mergeConfig: MergeConfiguration,
  weightsInitializer: Initializer? = GlorotInitializer(),
  biasesInitializer: Initializer? = null
) : Serializable {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The size of the merge output layer.
   */
  private val mergeOutputSize: Int = when (mergeConfig) {
    is AffineMerge -> mergeConfig.outputSize
    is BiaffineMerge -> mergeConfig.outputSize
    is ConcatFeedforwardMerge -> mergeConfig.outputSize
    is ConcatMerge -> this.inputSize + this.vectorSize
    is SumMerge, is ProductMerge, is AvgMerge -> {
      require(this.inputSize == this.vectorSize)
      this.inputSize
    }
    else -> throw RuntimeException("Invalid output merge configuration.")
  }

  /**
   * The merge network used to create the attention arrays of the [attentionParams].
   */
  val mergeNetwork = StackedLayersParameters(
    LayerInterface(
      sizes = listOf(this.inputSize, this.vectorSize)),
    LayerInterface(
      size = this.mergeOutputSize,
      activationFunction = (mergeConfig as? VariableOutputMergeConfig)?.activationFunction,
      connectionType = mergeConfig.type),
    weightsInitializer = weightsInitializer,
    biasesInitializer = biasesInitializer)

  /**
   * The parameters of the attention mechanism.
   */
  val attentionParams = AttentionMechanismLayerParameters(
    inputSize = this.mergeNetwork.outputSize,
    weightsInitializer = weightsInitializer)

  /**
   * The structure containing all the parameters of this model.
   */
  val params = PointerNetworkParameters(
    mergeParams = this.mergeNetwork,
    attentionParams = this.attentionParams)
}
