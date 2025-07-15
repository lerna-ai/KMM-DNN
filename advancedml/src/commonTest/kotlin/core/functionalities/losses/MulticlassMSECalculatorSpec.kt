/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.losses

import com.kotlinnlp.simplednn.core.functionalities.losses.MulticlassMSECalculator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class MulticlassMSECalculatorSpec : Spek({

  describe("a MulticlassMSECalculator") {

    val outputValues = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.7, 0.2, 0.1))
    val goldValues = DenseNDArrayFactory.arrayOf(doubleArrayOf(1.0, 0.0, 0.0, 0.0))

    context("calculateErrors") {
      val errors = MulticlassMSECalculator.calculateErrors(outputValues, goldValues)

      it("should calculate the expected errors") {
        assertTrue(DenseNDArrayFactory.arrayOf(doubleArrayOf(-1.0, 0.7, 0.2, 0.1)).equals(errors))
      }
    }
  }
})
