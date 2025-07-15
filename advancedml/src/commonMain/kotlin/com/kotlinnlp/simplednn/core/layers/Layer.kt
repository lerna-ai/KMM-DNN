/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.arrays.Norm1Array
import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.BaseRandom
import com.kotlinnlp.simplednn.core.layers.helpers.BackwardHelper
import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.helpers.ParamsErrorsCollector
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import java.lang.RuntimeException

/**
 * The Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property inputType the input array type (default Dense)
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout
 */
internal abstract class Layer<InputNDArrayType : NDArray<InputNDArrayType>>(
  val inputArray: AugmentedArray<InputNDArrayType>,
  val inputType: LayerType.Input, // = LayerType.Input.Dense,
  open val outputArray: AugmentedArray<DenseNDArray>,
  open val params: LayerParameters,
  val activationFunction: ActivationFunction? = null,
  val dropout: Double
) {

  /**
   * The probability to keep an output value (= no dropout).
   */
  private val p = 1.0 - this.dropout

  /**
   * Support to save the dropout mask.
   * It will contains values in the set {0.0, 1.0/[p]}.
   */
  private val dropoutMask: DenseNDArray = DenseNDArrayFactory.emptyArray(Shape(this.inputArray.size))

  /**
   * Support to save the original input in case of dropout.
   */
  private lateinit var nonDroppedInput: InputNDArrayType

  /**
   * Whether the dropout has been applied during the last forward.
   */
  private var dropoutApplied: Boolean = false

  /**
   * The helper which execute the forward
   */
  internal abstract val forwardHelper: ForwardHelper<InputNDArrayType>

  /**
   * The helper which execute the backward
   */
  internal abstract val backwardHelper: BackwardHelper<InputNDArrayType>

  /**
   * The helper which calculates the relevance
   */
  internal abstract val relevanceHelper: RelevanceHelper?

  /**
   * Whether the input is dense.
   */
  val denseInput: Boolean get() = this.inputType == LayerType.Input.Dense

  /**
   * Whether the input is sparse or binary sparse.
   */
  val sparseInput: Boolean get() = !this.denseInput

  /**
   * Set the values of the inputArray
   *
   * @param values the values to set into the inputArray
   */
  fun setInput(values: InputNDArrayType) = this.inputArray.assignValues(values)

  /**
   * Set the errors of the outputArray
   *
   * @param errors the errors to set into the outputArray
   */
  fun setErrors(errors: DenseNDArray) = this.outputArray.assignErrors(errors)

  /**
   * Set the relevance of the outputArray
   *
   * @param relevance the relevance to set into the outputArray
   */
  fun setOutputRelevance(relevance: Norm1Array<DenseNDArray>) {
    this.outputArray.assignRelevance(relevance.values)
  }

  /**
   * Set the params errors collector used by the [backwardHelper].
   *
   * @param c a collector of params errors
   */
  fun setParamsErrorsCollector(c: ParamsErrorsCollector) {
    this.backwardHelper.setParamsErrorsCollector(c)
  }

  /**
   * Return the params errors collector used by the [backwardHelper].
   */
  fun getParamsErrorsCollector(): ParamsErrorsCollector =
    this.backwardHelper.getParamsErrorsCollector()

  /**
   * Forward the input to the output combining it with the parameters.
   */
  fun forward() {

    this.dropoutApplied = this.applyDropout()

    this.forwardHelper.forward()
  }

  /**
   * Forward the input to the output combining it with the parameters, calculating its relevance respect of the output.
   *
   * @param contributions the support in which to save the contributions of the input respect to the output
   */
  fun forward(contributions: LayerParameters) {

    this.dropoutApplied = this.applyDropout()

    this.forwardHelper.forward(contributions)
  }

  /**
   * Calculate the relevance of the input respect to the output and assign it to the input array.
   *
   * @param contributions the contributions saved during the last forward
   */
  fun setInputRelevance(contributions: LayerParameters) {

    this.relevanceHelper
      ?.setInputRelevance(contributions = contributions)
      ?: throw RuntimeException("Relevance propagation not available.")
  }

  /**
   * Calculate the relevance of the input respect to the output and add it to the relevance of the input array
   * previously set.
   *
   * @param contributions the contributions saved during the last forward
   */
  fun addInputRelevance(contributions: LayerParameters) {
    this.relevanceHelper
      ?.addInputRelevance(contributions = contributions)
      ?: throw RuntimeException("Relevance propagation not available.")
  }

  /**
   * @param propagateToInput whether to propagate the errors to the input
   *
   * @return the params errors
   */
  fun backward(propagateToInput: Boolean): ParamsErrorsList = this.backwardHelper.backward(propagateToInput).also {

    if (this.dropoutApplied) {

      if (propagateToInput)
        this.addDropoutErrors()

      this.restoreInput()
    }
  }

  /**
   * Perform the multiplication of the output array by the derivative of its activated values.
   */
  fun applyOutputActivationDeriv() {

    if (this.outputArray.hasActivation) {

      val gY: DenseNDArray = this.outputArray.errors
      val gAct: DenseNDArray = this.outputArray.calculateActivationDeriv()

      if (gAct.isMatrix) // e.g. the Jacobian matrix in the Softmax function
        this.outputArray.assignErrorsByDot(gAct, gY)
      else
        this.outputArray.assignErrorsByProd(gAct, gY)
    }
  }

  /**
   * Apply the dropout of the input values, based on the probability [p].
   *
   * @return `true` if the dropout has been actually applied, otherwise `false`
   */
  private fun applyDropout(): Boolean {

    if (this.dropout > 0.0) {

      this.saveNonDroppedInput()

      this.dropoutMask
        .randomize(BaseRandom())
        .assignRoundInt(threshold = this.dropout)
        .assignDiv(this.p)

      this.inputArray.values.assignProd(this.dropoutMask)

      return true
    }

    return false
  }

  /**
   * Save the input values still not dropped into the [nonDroppedInput] support variable.
   */
  private fun saveNonDroppedInput() {

    if (::nonDroppedInput.isInitialized)
      this.nonDroppedInput.assignValues(this.inputArray.values)
    else
      this.nonDroppedInput = this.inputArray.values.copy()
  }

  /**
   * Add the dropout mask component to the input errors.
   */
  private fun addDropoutErrors() {
    this.inputArray.errors.assignProd(this.dropoutMask)
  }

  /**
   * Restore the original non-dropped input values.
   */
  private fun restoreInput() {
    this.inputArray.values.assignValues(this.nonDroppedInput)
  }
}
