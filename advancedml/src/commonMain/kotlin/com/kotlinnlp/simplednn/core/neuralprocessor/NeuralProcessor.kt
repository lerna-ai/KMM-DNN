/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.neuralprocessor

import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.utils.ItemsPool

/**
 * A generic neural processor.
 */
interface NeuralProcessor<
  in InputType : Any,
  out OutputType : Any,
  in ErrorsType : Any,
  out InputErrorsType : Any
  > : ItemsPool.IDItem {

  /**
   * The InputErrorsType to use in case the neural processor does not provide input errors.
   */
  object NoInputErrors

  /**
   * Whether to propagate the errors to the input during the [backward] (if supported).
   */
  val propagateToInput: Boolean

  /**
   * Execute the forward of the input to the output.
   *
   * @param input the input
   *
   * @return the output
   */
  fun forward(input: InputType): OutputType

  /**
   * Propagate the output errors with a stochastic gradient descent (SGD) algorithm.
   *
   * @param outputErrors the output errors
   */
  fun backward(outputErrors: ErrorsType)

  /**
   * Return the input errors of the last backward.
   * Before calling this method make sure that [propagateToInput] is enabled.
   *
   * @param copy whether to return by value or by reference (default true)
   *
   * @return the input errors
   */
  fun getInputErrors(copy: Boolean = true): InputErrorsType

  /**
   * Return the params errors of the last backward.
   *
   * @param copy whether the returned errors must be a copy or a reference (default true)
   *
   * @return the parameters errors
   */
  fun getParamsErrors(copy: Boolean = true): ParamsErrorsList

  /**
   * Back-propagate the [errors] through the network, accumulate the resulting params errors in the [optimizer] and
   * return the input errors.
   * Before calling this method make sure that [propagateToInput] is enabled.
   *
   * @param errors the output errors
   * @param optimizer the optimizer
   * @param copy whether the returned errors must be a copy or a reference (default false)
   *
   * @return the input errors
   */
  fun propagateErrors(errors: ErrorsType, optimizer: ParamsOptimizer, copy: Boolean = false): InputErrorsType {

    this.backward(errors)

    optimizer.accumulate(this.getParamsErrors(copy = copy))

    return getInputErrors(copy = copy)
  }
}
