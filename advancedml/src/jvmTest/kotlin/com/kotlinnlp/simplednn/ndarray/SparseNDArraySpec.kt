/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.ndarray

import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.ndarray.Indices
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayMask
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.SparseEntry
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.*

/**
 *
 */
class SparseNDArraySpec : Spek({

  describe("a SparseNDArray") {

    context("initialization") {

      context("indices out of bounds") {

        it("should raise an Exception") {

          assertFails {
            SparseNDArrayFactory.arrayOf(
              activeIndicesValues = arrayOf(
                Pair(Pair(0, 1), 0.1f),
                Pair(Pair(1, 0), 0.5f),
                Pair(Pair(1, 3), 0.1f),
                Pair(Pair(2, 2), 0.2f),
                Pair(Pair(3, 1), 0.3f)
              ),
              shape = Shape(4, 3))
          }
        }
      }
    }

    context("iteration") {

      context("2-dim array") {

        val array = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            Pair(Pair(0, 1), 0.1f),
            Pair(Pair(1, 0), 0.5f),
            Pair(Pair(1, 2), 0.1f),
            Pair(Pair(2, 2), 0.2f),
            Pair(Pair(3, 1), 0.3f)
          ),
          shape = Shape(4, 3))
        val iterator = array.iterator()

        it("should return the expected entry on the iteration 1") {
          assertEquals(Pair(Pair(1, 0), 0.5f), iterator.next())
        }

        it("should return the expected entry on the iteration 2") {
          assertEquals(Pair(Pair(0, 1), 0.1f), iterator.next())
        }

        it("should return the expected entry on the iteration 3") {
          assertEquals(Pair(Pair(3, 1), 0.3f), iterator.next())
        }

        it("should return the expected entry on the iteration 4") {
          assertEquals(Pair(Pair(1, 2), 0.1f), iterator.next())
        }

        it("should return the expected entry on the iteration 5") {
          assertEquals(Pair(Pair(2, 2), 0.2f), iterator.next())
        }

        it("should return false calling hasNext() on the last iteration") {
          assertFalse { iterator.hasNext() }
        }
      }
    }

    context("assignSumMerging()") {

      context("2-dim arrays") {

        val array1 = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            Pair(Pair(0, 1), 0.1f),
            Pair(Pair(1, 0), 0.5f),
            Pair(Pair(1, 2), 0.1f),
            Pair(Pair(2, 2), 0.2f),
            Pair(Pair(3, 1), 0.3f)
          ),
          shape = Shape(4, 3))

        val array2 = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            Pair(Pair(0, 1), 0.2f),
            Pair(Pair(1, 0), 0.1f),
            Pair(Pair(1, 2), 0.1f),
            Pair(Pair(2, 2), 0.5f),
            Pair(Pair(2, 1), 0.3f)
          ),
          shape = Shape(4, 3))

        val expectedArray = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            Pair(Pair(0, 1), 0.3f),
            Pair(Pair(1, 0), 0.6f),
            Pair(Pair(1, 2), 0.2f),
            Pair(Pair(2, 1), 0.3f),
            Pair(Pair(2, 2), 0.7f),
            Pair(Pair(3, 1), 0.3f)
          ),
          shape = Shape(4, 3))

        val res = array1.assignSum(array2)

        it("should return the same array") {
          assertSame(array1, res)
        }

        it("should contain the expected values") {
          assertTrue(expectedArray.equals(res))
        }
      }
    }

    context("math methods returning a new NDArray") {

      context("dot(denseArray) method") {

        val a1 = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            SparseEntry(Indices(0, 0), 0.5f),
            SparseEntry(Indices(1, 1), 0.5f)
          ),
          shape = Shape(3, 2))

        val a2 = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.2f, 0.9f),
          floatArrayOf(0.5f, 0.6f)
        ))

        val a3 = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.2f, 0.9f),
          floatArrayOf(0.5f, 0.6f),
          floatArrayOf(0.1f, 0.4f)
        ))

        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f, 0.45f),
          floatArrayOf(0.25f, 0.3f),
          floatArrayOf(0.0f, 0.0f)
        ))

        val res = a1.dot(a2)

        it("should throw an error with not compatible shapes") {
          assertFails { a1.dot(a3) }
        }

        it("should assign the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }
    }

    context("math methods in-place") {

      context("assignValues(denseArray, mask) method") {

        val aS = SparseNDArrayFactory.arrayOf(activeIndicesValues = arrayOf(), shape = Shape(4, 3))
        val aD = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.2f, 0.9f, 0.4f),
          floatArrayOf(0.5f, 0.6f, 0.1f),
          floatArrayOf(0.3f, 0.4f, 0.6f),
          floatArrayOf(0.1f, 0.0f, 0.1f)
        ))
        val bD = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.2f, 0.9f, 0.4f),
          floatArrayOf(0.5f, 0.6f, 0.1f)
        ))
        val mask = NDArrayMask(dim1 = intArrayOf(0, 1, 1, 3), dim2 = intArrayOf(1, 0, 2, 2))

        val expectedArray = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            SparseEntry(Indices(0, 1), 0.9f),
            SparseEntry(Indices(1, 0), 0.5f),
            SparseEntry(Indices(1, 2), 0.1f),
            SparseEntry(Indices(3, 2), 0.1f)
          ),
          shape = Shape(4, 3))

        val res = aS.assignValues(aD, mask = mask)

        it("should return the same DenseNDArray") {
          assertTrue { aS === res }
        }

        it("should assign the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }

        it("should throw an error with not compatible shapes") {
          assertFails { aS.assignValues(bD, mask = mask) }
        }
      }

      context("assignDot(sparseArray, denseArray) method") {

        val a = SparseNDArrayFactory.arrayOf(activeIndicesValues = arrayOf(), shape = Shape(4, 3))

        val aS = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            SparseEntry(Indices(0, 0), 0.5f),
            SparseEntry(Indices(2, 0), 1.0f)
          ),
          shape = Shape(4))

        val mS = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            SparseEntry(Indices(0, 0), 0.5f),
            SparseEntry(Indices(1, 1), 1.0f)
          ),
          shape = Shape(4, 2))

        val aD = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.2f, 0.9f, 0.5f)
        ))

        val b = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.2f, 0.9f, 0.4f),
          floatArrayOf(0.5f, 0.6f, 0.1f)
        ))

        val expectedArray = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            SparseEntry(Indices(0, 0), 0.1f),
            SparseEntry(Indices(0, 1), 0.45f),
            SparseEntry(Indices(0, 2), 0.25f),
            SparseEntry(Indices(2, 0), 0.2f),
            SparseEntry(Indices(2, 1), 0.9f),
            SparseEntry(Indices(2, 2), 0.5f)
          ),
          shape = Shape(4, 3))

        val res = a.assignDot(aS, aD)

        it("should return the same DenseNDArray") {
          assertTrue { a === res }
        }

        it("should assign the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }

        it("should throw an error with not compatible shapes") {
          assertFails { a.assignDot(aS, b) }
        }

        it("should throw an error with matrices") {
          assertFails { a.assignDot(mS, b) }
        }
      }
    }

    context("other math methods") {

      val array = SparseNDArrayFactory.arrayOf(
        activeIndicesValues = arrayOf(
          Pair(Pair(0, 1), 0.1f),
          Pair(Pair(1, 0), 0.5f),
          Pair(Pair(1, 2), 0.1f),
          Pair(Pair(2, 2), 0.2f),
          Pair(Pair(3, 1), 0.3f)
        ),
        shape = Shape(4, 3))

      context("sum() method") {

        it("should give the expected sum of its elements") {
          assertTrue { equals(1.2f, array.sum(), tolerance = 1.0e-10f) }
        }
      }

      context("max() method") {

        it("should have the expected max value") {
          assertEquals(0.5f, array.max())
        }
      }
    }
  }
})
