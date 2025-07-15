/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.activations

import com.kotlinnlp.simplednn.core.functionalities.activations.HardTanh
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */

class HardTanhSpec: Spek({

  describe("an HardTanh activation function") {

    val activationFunction = HardTanh
    val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 10.0, -1.0, -10.0))
    val activatedArray = activationFunction.f(array)

    context("f") {

      val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
          0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 1.0, -1.0, -1.0
      ))

      it("should return the expected values") {
        assertTrue { expectedArray.equals(activatedArray, tolerance = 1.0e-08) }
      }
    }

    context("dfOptimized") {

      val dfArray = activationFunction.dfOptimized(activatedArray)
      val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
          1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0
      ))

      it("should return the expected values") {
        assertTrue { expectedArray.equals(dfArray, tolerance = 1.0e-08) }
      }
    }
  }
})
