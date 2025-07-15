/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn.deepbirnn

import com.kotlinnlp.simplednn.core.neuralprocessor.NeuralProcessor
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNEncoder
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * Deep Bidirectional Recursive Neural Network Encoder.
 *
 * For convenience, this class exposes methods as if there was a single [BiRNN].
 * In this way, it is possible to use a [BiRNNEncoder] and a [DeepBiRNNEncoder] almost interchangeably.
 *
 * @property network the [DeepBiRNN] of this encoder
 * @param rnnDropout the probability of RNNs dropout
 * @param mergeDropout the probability of output merge dropout
 * @property propagateToInput whether to propagate the errors to the input during the [backward]
 * @property id an identification number useful to track a specific [DeepBiRNNEncoder]
 */
class DeepBiRNNEncoder<InputNDArrayType: NDArray<InputNDArrayType>>(
  val network: DeepBiRNN,
  rnnDropout: Double,
  mergeDropout: Double,
  override val propagateToInput: Boolean,
  override val id: Int = 0
): NeuralProcessor<
  List<InputNDArrayType>, // InputType
  List<DenseNDArray>, // OutputType
  List<DenseNDArray>, // ErrorsType
  List<DenseNDArray> // InputErrorsType
  > {

  /**
   * Deep Bidirectional Recursive Neural Network Encoder.
   *
   * For convenience, this class exposes methods as if there was a single [BiRNN].
   * In this way, it is possible to use a [BiRNNEncoder] and a [DeepBiRNNEncoder] almost interchangeably.
   *
   * @param network the [DeepBiRNN] of this encoder
   * @param dropout the probability of dropout, the same for the RNNs and the output merge (default 0.0)
   * @param propagateToInput whether to propagate the errors to the input during the [backward]
   * @param id an identification number useful to track a specific [DeepBiRNNEncoder]
   */
  constructor(
    network: DeepBiRNN,
    propagateToInput: Boolean,
    dropout: Double = 0.0,
    id: Int = 0
  ): this(
    network = network,
    rnnDropout = dropout,
    mergeDropout = dropout,
    propagateToInput = propagateToInput,
    id = id
  )

  /**
   * List of encoders for all the stacked [BiRNN] layers.
   */
  private val encoders = this.network.levels.mapIndexed { i, biRNN ->
    if (i == 0)
      BiRNNEncoder<InputNDArrayType>(
        network = biRNN,
        rnnDropout = rnnDropout,
        mergeDropout = mergeDropout,
        propagateToInput = this.propagateToInput)
    else
      BiRNNEncoder<DenseNDArray>(
        network = biRNN,
        rnnDropout = rnnDropout,
        mergeDropout = mergeDropout,
        propagateToInput = true)
  }

  /**
   * The Forward.
   *
   * @param input the input sequence
   *
   * @return the result of the forward
   */
  override fun forward(input: List<InputNDArrayType>): List<DenseNDArray> {

    var output: List<DenseNDArray>

    @Suppress("UNCHECKED_CAST")
    output = (this.encoders[0] as BiRNNEncoder<InputNDArrayType>).forward(input)

    for (i in 1 until this.encoders.size) {
      @Suppress("UNCHECKED_CAST")
      output = (this.encoders[i] as BiRNNEncoder<DenseNDArray>).forward(output)
    }

    return output
  }


  /**
   * Propagate the errors of the entire sequence.
   *
   * @param outputErrors the errors to propagate
   */
  override fun backward(outputErrors: List<DenseNDArray>) {

    var errors: List<DenseNDArray> = outputErrors

    this.encoders.reversed().forEach { encoder ->
      encoder.backward(errors)
      errors = encoder.getInputErrors(copy = false)
    }
  }

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the input sequence
   */
  override fun getInputErrors(copy: Boolean): List<DenseNDArray> =
    this.encoders.first().getInputErrors(copy = copy)

  /**
   * @param copy a Boolean indicating whether the returned errors must be a copy or a reference
   *
   * @return the errors of the DeepBiRNN parameters
   */
  override fun getParamsErrors(copy: Boolean) = this.encoders.flatMap { it.getParamsErrors(copy = copy) }

  /**
   * @param copy whether to return a copy of the arrays
   *
   * @return a pair containing the last output of the two RNNs (left-to-right, right-to-left).
   */
  fun getLastOutput(copy: Boolean): Pair<DenseNDArray, DenseNDArray> = this.encoders.last().getLastOutput(copy)

  /**
   * Propagate the errors of the last output of the two RNNs (left-to-right, right-to-left).
   *
   * @param leftToRightErrors the last output errors of the left-to-right network
   * @param rightToLeftErrors the last output errors of the right-to-left network
   */
  fun backwardLastOutput(leftToRightErrors: DenseNDArray, rightToLeftErrors: DenseNDArray) =
    this.encoders.last().backwardLastOutput(
      leftToRightErrors = leftToRightErrors,
      rightToLeftErrors = rightToLeftErrors)
}
