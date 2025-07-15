/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */

class TanhSpec : Spek({

  describe("a Tanh activation function") {

    val activationFunction = Tanh
    val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.1f, 0.01f, -0.1f, -0.01f, 1.0f, 10.0f, -1.0f, -10.0f))
    val activatedArray = activationFunction.f(array)

    context("f") {

      val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(
          0.0f, 0.099668f, 0.00999967f, -0.099668f, -0.00999967f, 0.76159416f, 1.0f, -0.76159416f, -1.0f
      ))

      it("should return the expected values") {
        assertTrue { expectedArray.equals(activatedArray, tolerance = 1.0e-08f) }
      }
    }

    context("dfOptimized") {

      val dfArray = activationFunction.dfOptimized(activatedArray)
      val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(
          1.0f, 0.99006629f, 0.99990001f, 0.99006629f, 0.999900007f, 0.41997434f, 8.2e-09f, 0.41997434f, 8.2e-09f
      ))

      it("should return the expected values") {
        assertTrue { expectedArray.equals(dfArray, tolerance = 1.0e-08f) }
      }
    }
  }
})
