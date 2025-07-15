/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.functionalities.activations

import com.kotlinnlp.simplednn.core.functionalities.activations.Softsign
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */

class SoftsignSpec : Spek({

  describe("a Softsign activation function") {

    val activationFunction = Softsign
    val array = DenseNDArrayFactory.arrayOf(doubleArrayOf(0.0, 0.1, 0.01, -0.1, -0.01, 1.0, 10.0, -1.0, -10.0))
    val activatedArray = activationFunction.f(array)

    context("f") {

      val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
        0.0, 0.09090909, 0.00990099, -0.09090909, -0.00990099, 0.5, 0.90909090, -0.5, -0.9090909
      ))

      it("should return the expected values") {
        assertTrue { expectedArray.equals(activatedArray, tolerance = 1.0e-08) }
      }
    }

    context("dfOptimized") {

      val dfArray = activationFunction.dfOptimized(activatedArray)
      val expectedArray = DenseNDArrayFactory.arrayOf(doubleArrayOf(
        1.0, 0.82644628, 0.98029605, 0.82644628, 0.98029605, 0.25, 0.00826446, 0.25, 0.00826446
      ))

      it("should return the expected values") {
        assertTrue { expectedArray.equals(dfArray, tolerance = 1.0e-08) }
      }
    }
  }
})
