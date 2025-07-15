/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.losses

import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 *
 */
class MSECalculatorSpec : Spek({

  describe("a MSECalculator") {

    val lossCalculator = MSECalculator()

    context("single output") {

      val outputValues = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.1f, 0.2f, 0.3f))
      val goldValues = DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, 0.2f, 0.1f, 0.0f))

      context("calculateErrors") {
        val errors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the expected errors") {
          assertTrue(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.3f, -0.1f, 0.1f, 0.3f)).equals(errors))
        }
      }

      context("calculateLoss") {
        val loss = lossCalculator.calculateLoss(outputValues, goldValues)

        it("should calculate the expected loss") {
          assertTrue(DenseNDArrayFactory.arrayOf(floatArrayOf(0.045f, 0.005f, 0.005f, 0.045f)).equals(loss))
        }
      }
    }

    context("output sequence") {

      val outputValuesSequence: List<DenseNDArray> = listOf(
        DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.1f, 0.2f, 0.3f)),
        DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, 0.1f, 0.4f, 0.7f))
      )
      val goldValuesSequence: List<DenseNDArray> = listOf(
        DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, 0.2f, 0.1f, 0.0f)),
        DenseNDArrayFactory.arrayOf(floatArrayOf(0.6f, 0.9f, 0.1f, 0.0f))
      )

      context("calculateErrors of a sequence of length 2") {

        val sequenceErrors: List<DenseNDArray> = lossCalculator.calculateErrors(outputValuesSequence, goldValuesSequence)

        it("should return an array of length 2") {
          assertEquals(2, sequenceErrors.size)
        }

        it("should calculate the expected errors of the first element") {
          assertTrue(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.3f, -0.1f, 0.1f, 0.3f)).equals(sequenceErrors[0]))
        }

        it("should calculate the expected errors of the second element") {
          assertTrue(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.1f, -0.8f, 0.3f, 0.7f)).equals(sequenceErrors[1]))
        }
      }

      context("calculateMeanLoss of a sequence of length 2") {
        val meanLoss = lossCalculator.calculateMeanLoss(outputValuesSequence, goldValuesSequence)

        it("should calculate the expected mean loss") {
          assertTrue(equals(0.089375f, meanLoss, tolerance = 1.0e-08f))
        }
      }
    }
  }
})
