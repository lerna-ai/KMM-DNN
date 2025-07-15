/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

import com.kotlinnlp.simplednn.core.functionalities.activations.SeLU
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */

class SeLUSpec: Spek({

  describe("a SeLU activation function") {

    val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.1f, 0.01f, -0.1f, -0.01f, 1.0f, 10.0f, -1.0f, -10.0f))

    context("default configuration") {

      val activationFunction = SeLU()
      val activatedArray = activationFunction.f(array)

      context("f") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(
            0.0f, 0.10507009f, 0.01050700f, -0.16730527f, -0.01749338f, 1.0507009f, 10.5070099f, -1.11133072f, -1.758019508f
        ))

        it("should return the expected values") {
          assertTrue { expectedArray.equals(activatedArray, tolerance = 1.0e-06f) }
        }
      }

      context("dfOptimized") {

        val dfArray = activationFunction.dfOptimized(activatedArray)
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(
            1.758099343f, 1.0507009f, 1.0507009f, 1.59079407f, 1.740605962f, 1.0507009f, 1.0507009f, 0.646768603f, 0.000079817586f
        ))

        it("should return the expected values") {
          assertTrue { expectedArray.equals(dfArray, tolerance = 1.0e-07f) }
        }
      }
    }
  }
})
