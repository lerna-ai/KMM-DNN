/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.functionalities.initializers.Initializer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array

/**
 * The parameters of a layer
 *
 * @property inputSize input size
 * @property outputSize output size
 * @param weightsInitializer the initializer of the weights (zeros if null)
 * @param biasesInitializer the initializer of the biases (zeros if null)
 */
@kotlinx.serialization.Serializable
abstract class LayerParameters(
  val inputSize: Int,
  val outputSize: Int,
  private val weightsInitializer: Initializer?,
  private val biasesInitializer: Initializer?
) {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   * The list of weights parameters.
   */
  abstract val weightsList: List<ParamsArray>

  /**
   * The list of biases parameters.
   */
  abstract val biasesList: List<ParamsArray>

  /**
   * Initialize the values of the parameters with the given [weightsInitializer] and [biasesInitializer].
   * If an initializer is null, its related parameters are initialized to zeros.
   *
   * Note: this method should be called into the 'init' block.
   */
  protected fun initialize() {

    this.weightsInitializer?.let { initializer ->
      this.weightsList.forEach { weight -> initializer.initialize(weight.values) }
    }

    this.biasesInitializer?.let { initializer ->
      this.biasesList.forEach { weight -> initializer.initialize(weight.values) }
    }
  }

  fun setParams(weights: List<D2Array<Float>>, biases: List<D2Array<Float>>) {
    this.weightsList.forEachIndexed{ index, weight -> weight.values.assignValues(DenseNDArray(weights[index])) }
    this.biasesList.forEachIndexed{ index, bias -> bias.values.assignValues(DenseNDArray(biases[index]))}
  }
}
