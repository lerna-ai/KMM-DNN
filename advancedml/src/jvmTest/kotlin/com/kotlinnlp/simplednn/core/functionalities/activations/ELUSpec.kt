/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

import com.kotlinnlp.simplednn.core.functionalities.activations.ELU
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */

class ELUSpec : Spek({

  describe("an ELU activation function") {

    val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.1f, 0.01f, -0.1f, -0.01f, 1.0f, 10.0f, -1.0f, -10.0f))

    context("alpha = 1.0") {

      val activationFunction = ELU(alpha = 1.0f)
      val activatedArray = activationFunction.f(array)

      context("f") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(
          0.0f, 0.1f, 0.01f, -0.095162582f, -0.009950166f, 1.0f, 10.0f, -0.632120559f, -0.9999546f
        ))

        it("should return the expected values") {
          assertTrue { expectedArray.equals(activatedArray, tolerance = 1.0e-07f) }
        }
      }

      context("dfOptimized") {

        val dfArray = activationFunction.dfOptimized(activatedArray)
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(
          1.0f, 1.0f, 1.0f, 0.904837418f, 0.990049834f, 1.0f, 1.0f, 0.367879441f, 0.0000454f
        ))

        it("should return the expected values") {
          assertTrue { expectedArray.equals(dfArray, tolerance = 1.0e-07f) }
        }
      }
    }
  }
})
