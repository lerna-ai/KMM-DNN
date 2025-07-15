/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.losses

import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import kotlin.math.exp

/**
 * Augmented Mean Squared Error calculator.
 *
 * @property pi pi (0.1 by default)
 * @property c c (10 by default)
 */
class AugmentedMSECalculator(val pi: Double = 0.1, val c: Double = 10.0) : LossCalculator {

  /**
   *
   */
  enum class InjectedErrorStrength(internal val weight: Double) {
    NONE(0.0),
    SOFT(0.01),
    MEDIUM(0.1),
    HARD(1.0)
  }

  /**
   *
   */
  var injectedErrorStrength = InjectedErrorStrength.NONE

  /**
   *
   */
  private val isLossPartitionDisabled: Boolean = pi == 0.0

  /**
   *
   */
  private val lossPartition: Double = 1.0 - pi

  /**
   * Calculate the loss between an output and its gold.
   *
   * @param output the output prediction
   * @param outputGold the expected output
   *
   * @return the loss within [output] and [outputGold]
   */
  override fun calculateLoss(output: DenseNDArray, outputGold: DenseNDArray): DenseNDArray {

    return if (this.isLossPartitionDisabled) {

      output.sub(outputGold).assignPow(2.0).assignProd(0.5)

    } else {
      val lossContribution = outputGold.sub(output).assignPow(2.0)
      val injectedContribution = output.prod(this.calculateRegularization()).assignPow(2.0)

      // 0.5 * ((1 - pi) * (g - o)^2 + pi * (o * reg)^2)
      lossContribution
        .assignProd(this.lossPartition)
        .assignSum(injectedContribution.assignProd(this.pi))
        .assignProd(0.5)
    }
  }

  /**
   * Calculate the errors between an output and its gold.
   *
   * @param output the output prediction
   * @param outputGold the expected output
   *
   * @return the derivative of the loss within [output] and [outputGold]
   */
  override fun calculateErrors(output: DenseNDArray, outputGold: DenseNDArray): DenseNDArray {

    return if (this.isLossPartitionDisabled) {

      output.sub(outputGold)

    } else {
      val injectedContribution = output.prod(this.calculateRegularization())

      // (1 - pi) * (o - g) + pi * (o * reg)
      output.sub(outputGold)
        .assignProd(this.lossPartition)
        .assignSum(injectedContribution.assignProd(this.pi))
    }
  }

  /**
   *
   */
  private fun calculateRegularization(): Double = 1.0 - exp(-c * this.injectedErrorStrength.weight)
}
