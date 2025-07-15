/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.losses

import com.kotlinnlp.simplednn.core.functionalities.losses.AugmentedMSECalculator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class AugmentedMSECalculatorSpec : Spek({

  describe("an AugmentedMSECalculator") {

    val outputValues = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.1f, 0.2f, 0.3f))
    val goldValues = DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, 0.2f, 0.1f, 0.0f))

    context("with loss partition disabled") {

      val lossCalculator = AugmentedMSECalculator(pi = 0.0f)

      context("calculateLoss") {

        val loss = lossCalculator.calculateLoss(outputValues, goldValues)

        it("should calculate the expected loss") {
          assertTrue(DenseNDArrayFactory.arrayOf(floatArrayOf(0.045f, 0.005f, 0.005f, 0.045f)).equals(loss))
        }
      }

      context("calculateErrors") {

        val errors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the expected errors") {
          assertTrue(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.3f, -0.1f, 0.1f, 0.3f)).equals(errors))
        }
      }
    }

    context("with none injected errors") {

      val lossCalculator = AugmentedMSECalculator()

      context("calculateLoss") {

        val loss = lossCalculator.calculateLoss(outputValues, goldValues)

        it("should calculate the expected loss") {
          assertTrue(DenseNDArrayFactory.arrayOf(floatArrayOf(0.0405f, 0.0045f, 0.0045f, 0.0405f)).equals(loss))
        }
      }

      context("calculateErrors") {

        val errors = lossCalculator.calculateErrors(outputValues, goldValues)

        it("should calculate the expected errors") {
          assertTrue(DenseNDArrayFactory.arrayOf(floatArrayOf(-0.27f, -0.09f, 0.09f, 0.27f)).equals(errors))
        }
      }
    }

    context("with hard injected errors") {

      val lossCalculator = AugmentedMSECalculator(pi = 0.9f, c = 15.0f)
      lossCalculator.injectedErrorStrength = AugmentedMSECalculator.InjectedErrorStrength.HARD

      context("calculateLoss") {

        val outputLoss = lossCalculator.calculateLoss(outputValues, goldValues)
        val expectedLoss = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0045f, 0.005f, 0.01849999f, 0.04499998f))

        it("should calculate the expected loss") {
          assertTrue(expectedLoss.equals(outputLoss, tolerance = 1.0e-08f))
        }
      }

      context("calculateErrors") {

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)
        val expectedErrors = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.03f, 0.07999997f, 0.18999995f, 0.29999992f))

        it("should calculate the expected errors") {
          assertTrue(expectedErrors.equals(outputErrors, tolerance = 1.0e-08f))
        }
      }
    }

    context("with medium injected errors") {

      val lossCalculator = AugmentedMSECalculator(pi = 0.9f, c = 15.0f)
      lossCalculator.injectedErrorStrength = AugmentedMSECalculator.InjectedErrorStrength.MEDIUM

      context("calculateLoss") {

        val outputLoss = lossCalculator.calculateLoss(outputValues, goldValues)
        val expectedLoss = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0045f, 0.00321587f, 0.01136348f, 0.02894283f))

        it("should calculate the expected loss") {
          assertTrue(expectedLoss.equals(outputLoss, tolerance = 1.0e-08f))
        }
      }

      context("calculateErrors") {

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)
        val expectedErrors = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.03f, 0.05991829f, 0.14983657f, 0.23975486f))

        it("should calculate the expected errors") {
          assertTrue(expectedErrors.equals(outputErrors, tolerance = 1.0e-08f))
        }
      }
    }

    context("with low injected errors") {

      val lossCalculator = AugmentedMSECalculator(pi = 0.9f, c = 15.0f)
      lossCalculator.injectedErrorStrength = AugmentedMSECalculator.InjectedErrorStrength.SOFT

      context("calculateLoss") {

        val outputLoss = lossCalculator.calculateLoss(outputValues, goldValues)
        val expectedLoss = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0045f, 0.00058731f, 0.00084924f, 0.00528579f))

        it("should calculate the expected loss") {
          assertTrue(expectedLoss.equals(outputLoss, tolerance = 1.0e-08f))
        }
      }

      context("calculateErrors") {

        val outputErrors = lossCalculator.calculateErrors(outputValues, goldValues)
        val expectedErrors = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.03f, 0.00253628f, 0.03507256f, 0.06760885f))

        it("should calculate the expected errors") {
          assertTrue(expectedErrors.equals(outputErrors, tolerance = 1.0e-08f))
        }
      }
    }
  }
})
