/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

import com.kotlinnlp.simplednn.core.functionalities.activations.Softmax
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 *
 */
class SoftmaxSpec : Spek({

  describe("a Softmax activation function") {

    val activationFunction = Softmax()
    val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.1f, 0.01f, -0.1f, -0.01f, 1.0f, 10.0f, -1.0f, -10.0f))
    val activatedArray = activationFunction.f(array)

    context("f") {

      val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(
        4.53832e-05f, 5.01562e-05f, 4.58394e-05f, 4.10645e-05f,
        4.49317e-05f, 1.233645e-04f, 9.996325629e-01f, 1.66956e-05f, 2.1e-09f
      ))

      it("should have 1.0 as the sum its element") {
        assertEquals(1.0f, activatedArray.sum(), 1.0e-06f)
      }

      it("should return the expected values") {
        assertTrue(expectedArray.equals(activatedArray, tolerance = 1.0e-07f))
      }
    }

    context("dfOptimized") {

      context("returning a new NDArray as output") {

        val dfArray = activationFunction.dfOptimized(activatedArray)

        it("should return the expected values") {
          assertTrue(activatedArray.onesLike().equals(dfArray, tolerance = 1.0e-08f))
        }
      }

      context("assigning the results to an output array") {

        val outDfArray = DenseNDArrayFactory.emptyArray(array.shape)
        activationFunction.dfOptimized(activatedArray, outDfArray)

        it("should assign the expected values") {
          assertTrue(activatedArray.onesLike().equals(outDfArray, tolerance = 1.0e-08f))
        }
      }
    }
  }
})
