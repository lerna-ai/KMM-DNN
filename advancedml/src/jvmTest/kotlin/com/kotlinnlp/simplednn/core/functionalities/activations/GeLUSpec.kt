/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.activations

import com.kotlinnlp.simplednn.core.functionalities.activations.GeLU
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */

class GeLUSpec: Spek({

  describe("a GeLU activation function") {

    val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.1f, 0.01f, -0.1f, -0.01f, 1.0f, 10.0f, -1.0f, -10.0f))

    context("default configuration") {

      val activationFunction = GeLU
      val activatedArray = activationFunction.f(array)

      context("f") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(
          0.0f, 0.053983f, 0.00504f, -0.046017f, -0.00496f, 0.841192f, 10.0f, -0.158808f, 0.0f
        ))

        it("should return the expected values") {
          assertTrue { expectedArray.equals(activatedArray, tolerance = 1.0e-06f) }
        }
      }

      context("df") {

        val dfArray = activationFunction.df(array)
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(
          0.5f, 0.579522f, 0.507979f, 0.420478f, 0.492021f, 1.082964f, 1.0f, -0.082964f, 0.0f
        ))

        it("should return the expected values") {
          assertTrue { expectedArray.equals(dfArray, tolerance = 1.0e-06f) }
        }
      }
    }
  }
})
