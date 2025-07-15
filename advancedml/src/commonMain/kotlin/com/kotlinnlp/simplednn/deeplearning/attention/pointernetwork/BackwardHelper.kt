/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.attention.pointernetwork

import com.kotlinnlp.simplednn.core.layers.helpers.ParamsErrorsCollector
import com.kotlinnlp.simplednn.core.layers.models.attention.attentionmechanism.AttentionMechanismLayer
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The backward helper of the [PointerNetworkProcessor].
 *
 * @property networkProcessor the attentive recurrent network of this helper
 */
class BackwardHelper(private val networkProcessor: PointerNetworkProcessor) {

  /**
   * The list of errors of the input sequence.
   */
  internal lateinit var inputSequenceErrors: List<DenseNDArray>
    private set

  /**
   * The list of errors of the input vectors.
   */
  internal lateinit var vectorsErrors: List<DenseNDArray>
    private set

  /**
   * The index of the current state (the backward processes the states in inverted order).
   */
  private var stateIndex: Int = 0

  /**
   * The errors of the attention parameters which will be filled at each backward.
   */
  private var attentionParamsErrorsCollector = ParamsErrorsCollector()

  /**
   * The params errors accumulator of the attention structure
   */
  private var attentionErrorsAccumulator = ParamsErrorsAccumulator()

  /**
   * Perform the back-propagation from the output errors.
   *
   * @param outputErrors the errors to propagate
   */
  fun backward(outputErrors: List<DenseNDArray>) {

    this.initBackward()

    outputErrors.indices.reversed().forEach { stateIndex ->

      this.stateIndex = stateIndex

      this.backwardStep(outputErrors[stateIndex])
    }
  }

  /**
   * @param copy a Boolean indicating if the returned errors must be a copy or a reference
   *
   * @return the params errors of the [networkProcessor]
   */
  fun getParamsErrors(copy: Boolean = true) =
    this.networkProcessor.mergeProcessor.getParamsErrors(copy = copy) +
      this.attentionErrorsAccumulator.getParamsErrors(copy = copy)

  /**
   * A single step of backward.
   *
   * @param outputErrors the errors of a single output array
   */
  private fun backwardStep(outputErrors: DenseNDArray) {

    val attentionArraysErrors: List<DenseNDArray> = this.backwardAttentionScores(outputErrors)
    val vectorErrors: DenseNDArray = this.backwardAttentionArrays(attentionArraysErrors)

    this.vectorsErrors[this.stateIndex].assignValues(vectorErrors)
  }

  /**
   * @param outputErrors the errors of a single output array
   *
   * @return the errors of the attention arrays
   */
  private fun backwardAttentionScores(outputErrors: DenseNDArray): List<DenseNDArray> {

    val attentionMechanism: AttentionMechanismLayer = this.networkProcessor.usedAttentionMechanisms[this.stateIndex]

    attentionMechanism.setErrors(outputErrors)
    attentionMechanism.setParamsErrorsCollector(this.attentionParamsErrorsCollector)

    this.attentionErrorsAccumulator.accumulate(attentionMechanism.backward(propagateToInput = true))

    return attentionMechanism.inputArrays.map { it.errors }
  }

  /**
   * @param outputErrors the errors of the attention arrays
   *
   * @return the errors of the input vector
   */
  private fun backwardAttentionArrays(outputErrors: List<DenseNDArray>): DenseNDArray {

    val contextVectorErrorsSum = DenseNDArrayFactory.zeros(Shape(this.networkProcessor.model.inputSize))

    this.networkProcessor.mergeProcessor.backward(outputErrors)

    this.networkProcessor.mergeProcessor.getInputsErrors(copy = true).forEachIndexed { i, errors ->

      val inputErrors = errors[0]
      val contextVectorErrors = errors[1]

      contextVectorErrorsSum.assignSum(contextVectorErrors)
      this.inputSequenceErrors[i].assignSum(inputErrors)
    }

    return contextVectorErrorsSum
  }

  /**
   * Initialize the structures used during a backward.
   */
  private fun initBackward() {

    this.initInputSequenceErrors()
    this.initVectorsErrors()

    this.attentionErrorsAccumulator.clear()
  }

  /**
   * Initialize the [inputSequenceErrors] with arrays of zeros (an amount equal to the size of the current input
   * sequence).
   */
  private fun initInputSequenceErrors() {
    this.inputSequenceErrors = List(
      size = this.networkProcessor.inputSequence.size,
      init = { DenseNDArrayFactory.zeros(Shape(this.networkProcessor.model.inputSize)) })
  }

  /**
   * Initialize the [vectorsErrors] with arrays of zeros (an amount equal to the size of the number of
   * performed forward).
   */
  private fun initVectorsErrors() {
    this.vectorsErrors = List(
      size = this.networkProcessor.forwardCount,
      init = { DenseNDArrayFactory.zeros(Shape(this.networkProcessor.model.vectorSize)) })
  }
}
