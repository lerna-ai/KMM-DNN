/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.feedforward.highway

import com.kotlinnlp.simplednn.core.functionalities.activations.ActivationFunction
import com.kotlinnlp.simplednn.core.arrays.AugmentedArray
import com.kotlinnlp.simplednn.core.functionalities.activations.Sigmoid
import com.kotlinnlp.simplednn.core.layers.*
import com.kotlinnlp.simplednn.core.layers.helpers.RelevanceHelper
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray

/**
 * The Highway Layer Structure.
 *
 * @property inputArray the input array of the layer
 * @property inputType the type of the input arrays (default Dense)
 * @property outputArray the output array of the layer
 * @property params the parameters which connect the input to the output
 * @property activationFunction the activation function of the layer
 * @property dropout the probability of dropout
 */
internal class HighwayLayer<InputNDArrayType : NDArray<InputNDArrayType>>(
  inputArray: AugmentedArray<InputNDArrayType>,
  inputType: LayerType.Input,
  outputArray: AugmentedArray<DenseNDArray>,
  override val params: HighwayLayerParameters,
  activationFunction: ActivationFunction? = null,
  dropout: Double
) : Layer<InputNDArrayType>(
  inputArray = inputArray,
  inputType = inputType,
  outputArray = outputArray,
  params = params,
  activationFunction = activationFunction,
  dropout = dropout
) {

  /**
   * The input unit.
   */
  val inputUnit = AugmentedArray.zeros(outputArray.size)

  /**
   * The transform gate.
   */
  val transformGate = AugmentedArray.zeros(outputArray.size)

  /**
   * The helper which executes the forward
   */
  override val forwardHelper = HighwayForwardHelper(layer = this)

  /**
   * The helper which executes the backward
   */
  override val backwardHelper = HighwayBackwardHelper(layer = this)

  /**
   * The helper which calculates the relevance
   */
  override val relevanceHelper: RelevanceHelper? = null

  /**
   * Initialization: set the activation function of the outputArray
   */
  init {

    require(inputArray.size == outputArray.size) {
      "The Highway layer requires the input size to be equal to the output size."
    }

    this.transformGate.setActivation(Sigmoid)

    if (activationFunction != null) {
      this.inputUnit.setActivation(activationFunction)
    }
  }
}
