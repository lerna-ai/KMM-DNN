/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn

import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNNUtils
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 *
 */
class BiRNNUtilsSpec : Spek({

  describe("a BiRNNUtils") {

    val array1 = listOf(
      DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.8f, 0.8f, -1.0f, -0.7f)),
      DenseNDArrayFactory.arrayOf(floatArrayOf(-0.7f, -0.8f, 0.2f, -0.7f, 0.7f)),
      DenseNDArrayFactory.arrayOf(floatArrayOf(-0.9f, 0.9f, 0.7f, -0.5f, 0.5f)),
      DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, -0.1f, 0.5f, -0.2f, -0.8f)),
      DenseNDArrayFactory.arrayOf(floatArrayOf(-0.6f, 0.6f, 0.8f, -0.1f, -0.3f)))

    val array2 = listOf(
      DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, -0.6f, -1.0f, -0.1f, -0.4f)),
      DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, -0.9f, 0.0f, 0.8f, 0.3f)),
      DenseNDArrayFactory.arrayOf(floatArrayOf(-0.3f, -0.9f, 0.3f, 1.0f, -0.2f)),
      DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, 0.2f, 0.3f, -0.4f, -0.6f)),
      DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2f, 0.5f, -0.2f, -0.9f, 0.4f)))

    context("sumBidirectionalErrors") {

      val result: List<DenseNDArray> = BiRNNUtils.sumBidirectionalErrors(array1, array2)

      val expectedResult = listOf(
        DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2f, 1.3f, 0.6f, -1.9f, -0.3f)),
        DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, -0.6f, 0.5f, -1.1f, 0.1f)),
        DenseNDArrayFactory.arrayOf(floatArrayOf(-1.2f, 0.0f, 1.0f, 0.5f, 0.3f)),
        DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, -1.0f, 0.5f, 0.6f, -0.5f)),
        DenseNDArrayFactory.arrayOf(floatArrayOf(-0.5f, 0.0f, -0.2f, -0.2f, -0.7f))
      )

      it("should return an array of the expected size") {
        assertEquals(expectedResult.size, result.size)
      }

      it("should return an array with elements of the expected shape") {
        assertTrue { result.all{ it.shape == Shape(5, 1) } }
      }

      it("should return an array with elements of same shape of the expected values") {
        assertTrue { expectedResult.zip(result).all { (a, b) -> a.shape == b.shape } }
      }

      it("should return the pre-calculated values") {
        assertEquals(expectedResult, result)
      }
    }

    context("splitErrors") {

      val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.8f, 0.8f, -1.0f, -0.7f, 0.1f, -0.6f, -1.0f, -0.1f, -0.4f))

      val (result1f, result2f) = BiRNNUtils.splitErrors(array)

      val expectedResult1 = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.8f, 0.8f, -1.0f, -0.7f))
      val expectedResult2 = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, -0.6f, -1.0f, -0.1f, -0.4f))

      it("should return the pre-calculated values on ") {
        assertTrue { expectedResult1.equals(result1f) }
        assertTrue { expectedResult2.equals(result2f) }
      }
    }
  }
})
