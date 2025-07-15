/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.ndarray

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.simplemath.concatVectorsV
import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.exp
import com.kotlinnlp.simplednn.simplemath.ndarray.Indices
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayMask
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.SparseEntry
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArrayFactory
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.*

/**
 *
 */
class DenseNDArraySpec : Spek({

  describe("a DenseNDArray") {

    context("class factory methods") {

      context("arrayOf()") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))

        it("should have the expected length") {
          assertEquals(4, array.length)
        }

        it("should have the expected last index") {
          assertEquals(3, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(4, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(1, array.columns)
        }

        it("should contain the expected value at index 0") {
          assertEquals(0.1f, array[0])
        }

        it("should contain the expected value at index 1") {
          assertEquals(0.2f, array[1])
        }

        it("should contain the expected value at index 2") {
          assertEquals(0.3f, array[2])
        }

        it("should contain the expected value at index 3") {
          assertEquals(0.0f, array[3])
        }
      }

      context("fromRows()") {

        val matrix = DenseNDArrayFactory.fromRows(listOf(
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f)),
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, 0.4f)),
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, 0.6f))
        ))

        it("should have the expected shape") {
          assertEquals(Shape(3, 2), matrix.shape)
        }

        it("should have the expected values") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(0.1f, 0.2f),
              floatArrayOf(0.3f, 0.4f),
              floatArrayOf(0.5f, 0.6f)
            )).equals(matrix, tolerance = 0.001f)
          }
        }

        it("should raise an exception if the shapes are not compatible") {
          assertFailsWith<IllegalArgumentException> {
            DenseNDArrayFactory.fromRows(listOf(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f)),
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, 0.4f)),
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, 0.8f, 0.9f))
            ))
          }
        }
      }

      context("fromColumns()") {

        val matrix = DenseNDArrayFactory.fromColumns(listOf(
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f)),
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, 0.4f)),
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, 0.6f))
        ))

        it("should have the expected shape") {
          assertEquals(Shape(2, 3), matrix.shape)
        }

        it("should have the expected values") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(0.1f, 0.3f, 0.5f),
              floatArrayOf(0.2f, 0.4f, 0.6f)
            )).equals(matrix, tolerance = 0.001f)
          }
        }

        it("should raise an exception if the shapes are not compatible") {
          assertFailsWith<IllegalArgumentException> {
            DenseNDArrayFactory.fromColumns(listOf(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f)),
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, 0.4f)),
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, 0.8f, 0.9f))
            ))
          }
        }
      }

      context("zeros()") {

        val array = DenseNDArrayFactory.zeros(Shape(2, 3))

        it("should have the expected length") {
          assertEquals(6, array.length)
        }

        it("should have the expected last index") {
          assertEquals(5, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(2, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(3, array.columns)
        }

        it("should be filled with zeros") {
          (0 until array.length).forEach { assertEquals(0.0f, array[it]) }
        }
      }

      context("ones()") {

        val array = DenseNDArrayFactory.ones(Shape(2, 3))

        it("should have the expected length") {
          assertEquals(6, array.length)
        }

        it("should have the expected last index") {
          assertEquals(5, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(2, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(3, array.columns)
        }

        it("should be filled with ones") {
          (0 until array.length).forEach { assertEquals(1.0f, array[it]) }
        }
      }

      context("eye()") {

        val array = DenseNDArrayFactory.eye(4)

        it("should have the expected length") {
          assertEquals(16, array.length)
        }

        it("should have the expected last index") {
          assertEquals(15, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(4, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(4, array.columns)
        }

        it("should be filled with the expected values") {
          assertTrue {
            (0 until array.rows).all { i ->
              (0 until array.columns).all { j ->
                if (i == j) array[i, j] == 1.0f else array[i, j] == 0.0f
              }
            }
          }
        }
      }

      context("fill()") {

        val array = DenseNDArrayFactory.fill(shape = Shape(2, 3), value = 0.35f)

        it("should have the expected length") {
          assertEquals(6, array.length)
        }

        it("should have the expected last index") {
          assertEquals(5, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(2, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(3, array.columns)
        }

        it("should be filled with the expected value") {
          (0 until array.length).forEach { assertEquals(0.35f, array[it]) }
        }
      }

      context("emptyArray()") {

        val array = DenseNDArrayFactory.emptyArray(Shape(3, 2))

        it("should have the expected length") {
          assertEquals(6, array.length)
        }

        it("should have the expected last index") {
          assertEquals(5, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(3, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(2, array.columns)
        }
      }

      context("oneHotEncoder()") {

        val array = DenseNDArrayFactory.oneHotEncoder(length = 4, oneAt = 2)

        it("should have the expected length") {
          assertEquals(4, array.length)
        }

        it("should have the expected last index") {
          assertEquals(3, array.lastIndex)
        }

        it("should have the expected length") {
          assertEquals(4, array.length)
        }

        it("should be a column vector") {
          assertEquals(1, array.columns)
        }

        it("should have the expected values") {
          assertTrue { DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 1.0f, 0.0f)).equals(array) }
        }
      }

      context("random()") {

        val array = DenseNDArrayFactory.random(shape = Shape(216, 648), from = 0.5f, to = 0.89f)

        it("should have the expected length") {
          assertEquals(139968, array.length)
        }

        it("should have the expected last index") {
          assertEquals(139967, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(216, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(648, array.columns)
        }

        it("should contain values within the expected range") {
          (0 until array.length).forEach { i ->
            val value = array[i]
            assertTrue { value >= 0.5f && value < 0.89f }
          }
        }
      }

      context("exp()") {

        val power = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f, 0.2f),
          floatArrayOf(0.3f, 0.0f)
        ))
        val array = exp(power)

        it("should have the expected length") {
          assertEquals(4, array.length)
        }

        it("should have the expected last index") {
          assertEquals(3, array.lastIndex)
        }

        it("should have the expected number of rows") {
          assertEquals(2, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(2, array.columns)
        }

        it("should contain the expected value at index 0") {
          assertTrue { equals(1.105171f, array[0, 0], tolerance = 1.0e-06f) }
        }

        it("should contain the expected value at index 1") {
          assertTrue { equals(1.221403f, array[0, 1], tolerance = 1.0e-06f) }
        }

        it("should contain the expected value at index 2") {
          assertTrue { equals(1.349859f, array[1, 0], tolerance = 1.0e-06f) }
        }

        it("should contain the expected value at index 3") {
          assertTrue { equals(1.0f, array[1, 1], tolerance = 1.0e-06f) }
        }
      }
    }

    context("equality with tolerance") {

      val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.123f, 0.234f, 0.345f, 0.012f))

      context("comparison with different types") {

        val arrayToCompare = SparseNDArrayFactory.arrayOf(
          activeIndicesValues = arrayOf(
            SparseEntry(Indices(0, 0), 0.123f),
            SparseEntry(Indices(1, 0), 0.234f),
            SparseEntry(Indices(2, 0), 0.345f),
            SparseEntry(Indices(3, 0), 0.012f)
          ),
          shape = Shape(4))

        it("should return false") {
          assertFalse { array.equals(arrayToCompare, tolerance = 1.0e-03f) }
        }
      }

      context("comparison within the tolerance") {

        val arrayToCompare = DenseNDArrayFactory.arrayOf(
          floatArrayOf(0.123f, 0.234f, 0.345f, 0.012000001f))

        it("should result equal with a large tolerance") {
          assertTrue { array.equals(arrayToCompare, tolerance=1.0e-03f) }
        }

        it("should result equal with a strict tolerance") {
          assertTrue { array.equals(arrayToCompare, tolerance=1.0e-08f) }
        }
      }

      context("comparison out of tolerance") {

        val arrayToCompare = DenseNDArrayFactory.arrayOf(
          floatArrayOf(0.12303f, 0.23403f, 0.34503f, 0.01203f))

        it("should result not equal") {
          assertFalse { array.equals(arrayToCompare, tolerance=1.0e-05f) }
        }
      }
    }

    context("initialization through a float array of 4 elements") {

      val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))

      context("properties") {

        it("should be a vector") {
          assertTrue { array.isVector }
        }

        it("should not be a matrix") {
          assertFalse { array.isMatrix }
        }

        it("should have the expected length") {
          assertEquals(4, array.length)
        }

        it("should have the expected number of rows") {
          assertEquals(4, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(1, array.columns)
        }

        it("should have the expected shape") {
          assertEquals(Shape(4), array.shape)
        }
      }

      context("generic methods") {

        it("should be equal to itself") {
          assertTrue { array.equals(array) }
        }

        it("should be equal to its copy") {
          assertTrue { array.equals(array.copy()) }
        }
      }

      context("getRange() method") {

        val a = array.getRange(0, 3)
        val b = array.getRange(2, 4)

        it("should return a range of the expected length") {
          assertEquals(3, a.length)
        }

        it("should return the expected range (0f, 3f)") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f)).equals(a)
          }
        }

        it("should return the expected range (2f, 4f)") {
          assertTrue {
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, 0.0f)).equals(b)
          }
        }

        it("should raise an IndexOutOfBoundsException requesting for a range out of bounds") {
          assertFailsWith<IllegalStateException> {
            array.getRange(2, 6)
          }
        }
      }

      context("transpose") {

        val transposedArray = array.t

        it("should give a transposed array with the expected shape") {
          assertEquals(Shape(1, 4), transposedArray.shape)
        }

        it("should give a transposed array with the expected values") {
          assertEquals(transposedArray[2], 0.3f)
        }
      }
    }

    context("isOneHotEncoder() method") {

      val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
      val oneHotEncoder = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 1.0f, 0.0f))
      val oneHotEncoderFloat = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 1.0f, 1.0f, 0.0f))
      val oneHotEncoderFake = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.1f, 0.0f, 0.0f))
      val array2 = DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f),
        floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f)
      ))

      it("should return false on a random array") {
        assertFalse { array.isOneHotEncoder }
      }

      it("should return false on a 2-dim array") {
        assertFalse { array2.isOneHotEncoder }
      }

      it("should return false on an array with one element equal to 0.1") {
        assertFalse { oneHotEncoderFake.isOneHotEncoder }
      }

      it("should return false on an array with two elements equal to 1.0") {
        assertFalse { oneHotEncoderFloat.isOneHotEncoder }
      }

      it("should return true on an array with one element equal to 1.0") {
        assertTrue { oneHotEncoder.isOneHotEncoder }
      }
    }

    context("math methods returning a new NDArray") {

      val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
      val a = DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.3f, 0.5f, 0.7f))
      val n = 0.9f

      context("sum(number) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(1.0f, 1.1f, 1.2f, 0.9f))
        val res = array.sum(n)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("sum(array) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, 0.5f, 0.8f, 0.7f))
        val res = array.sum(a)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("sumByRows(array) method") {

        val matrix = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f),
          floatArrayOf(0.4f, 0.5f, 0.7f, 0.9f)
        ))
        val expectedRes = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.2f, 0.4f, 0.6f, 0.0f),
          floatArrayOf(0.5f, 0.7f, 1.0f, 0.9f)
        ))
        val res = matrix.sumByRows(array)

        it("should return a new DenseNDArray") {
          assertFalse { matrix === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedRes, tolerance = 1.0e-04f) }
        }
      }


      context("sumByColumns(array) method") {

        val sumArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f))
        val matrix = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f),
          floatArrayOf(0.4f, 0.5f, 0.7f, 0.9f)
        ))
        val expectedRes = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.2f, 0.3f, 0.4f, 0.1f),
          floatArrayOf(0.6f, 0.7f, 0.9f, 1.1f)
        ))
        val res = matrix.sumByColumns(sumArray)

        it("should return a new DenseNDArray") {
          assertFalse { matrix === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedRes, tolerance = 1.0e-04f) }
        }
      }

      context("sub(number) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.8f, -0.7f, -0.6f, -0.9f))
        val res = array.sub(n)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("sub(array) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.3f, -0.1f, -0.2f, -0.7f))
        val res = array.sub(a)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should assign the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("reverseSub(number) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.8f, 0.7f, 0.6f, 0.9f))
        val res = array.reverseSub(n)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("dot(array) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.04f, 0.03f, 0.05f, 0.07f),
          floatArrayOf(0.08f, 0.06f, 0.1f, 0.14f),
          floatArrayOf(0.12f, 0.09f, 0.15f, 0.21f),
          floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)
        ))
        val res = array.dot(a.t)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should throw an error with not compatible shapes") {
          assertFails { array.dot(a) }
        }

        it("should assign the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("dotLeftMasked(array, mask) method") {

        val a1 = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.5f, 0.3f),
          floatArrayOf(1.0f, 0.5f),
          floatArrayOf(0.7f, 0.6f)
        ))
        val a2 = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.2f, 0.9f),
          floatArrayOf(0.5f, 0.6f)
        ))
        val expected = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f, 0.45f),
          floatArrayOf(0.25f, 0.3f),
          floatArrayOf(0.0f, 0.0f)
        ))
        val res = a1.dotLeftMasked(a2, mask = NDArrayMask(dim1 = intArrayOf(0, 1), dim2 = intArrayOf(0, 1)))

        it("should throw an error with not compatible shapes") {
          val a3 = DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.7f, 0.5f),
            floatArrayOf(0.3f, 0.2f),
            floatArrayOf(0.3f, 0.5f),
            floatArrayOf(0.7f, 0.5f)
          ))
          assertFails { array.assignDot(a1, a3) }
        }

        it("should assign the expected values") {
          assertTrue { expected.equals(res, tolerance = 1.0e-04f) }
        }
      }

      context("dotRightMasked(array, mask) method") {

        val a1 = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.5f, 0.3f),
          floatArrayOf(1.0f, 0.5f),
          floatArrayOf(0.7f, 0.6f)
        ))
        val a2 = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.2f, 0.9f),
          floatArrayOf(0.5f, 0.6f)
        ))
        val expected = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f, 0.18f),
          floatArrayOf(0.2f, 0.3f),
          floatArrayOf(0.14f, 0.36f)
        ))
        val res = a1.dotRightMasked(a2, mask = NDArrayMask(dim1 = intArrayOf(0, 1), dim2 = intArrayOf(0, 1)))

        it("should throw an error with not compatible shapes") {
          val a3 = DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.7f, 0.5f),
            floatArrayOf(0.3f, 0.2f),
            floatArrayOf(0.3f, 0.5f),
            floatArrayOf(0.7f, 0.5f)
          ))
          assertFails { array.assignDot(a1, a3) }
        }

        it("should assign the expected values") {
          assertTrue { expected.equals(res, tolerance = 1.0e-04f) }
        }
      }

      context("prod(number) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.09f, 0.18f, 0.27f, 0.0f))
        val res = array.prod(n)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("prod(array) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.04f, 0.06f, 0.15f, 0.0f))
        val res = array.prod(a)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("matrix.prod(colVector) method") {

        val matrix = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f),
          floatArrayOf(0.4f, 0.5f, 0.7f, 0.9f)
        ))
        val colVector = DenseNDArrayFactory.arrayOf(floatArrayOf(0.2f, 0.3f))
        val expectedMatrix = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.02f, 0.04f, 0.06f, 0.0f),
          floatArrayOf(0.12f, 0.15f, 0.21f, 0.27f)
        ))
        val res = matrix.prod(colVector)

        it("should return a new DenseNDArray") {
          assertFalse { matrix === res }
        }

        it("should return the expected values "+res) {
          assertTrue { res.equals(expectedMatrix, tolerance = 1.0e-04f) }
        }
      }

      context("div(number) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1111f, 0.2222f, 0.3333f, 0.0f))
        val res = array.div(n)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("div(array) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.25f, 0.6667f, 0.6f, 0.0f))
        val res = array.div(a)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("roundInt(threshold) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 1.0f, 1.0f, 0.0f))
        val res = array.roundInt(threshold = 0.2f)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("avg() method") {

        it("should return the expected average") {
          assertTrue { equals(0.15f, array.avg(), tolerance = 1.0e-08f) }
        }
      }

      context("sign() method") {

        val signedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.1f, 0.0f, 0.7f, -0.6f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(-1.0f, 0.0f, 1.0f, -1.0f))
        val res = signedArray.sign()

        it("should return a new DenseNDArray") {
          assertFalse { signedArray === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("nonZeroSign() method") {

        val signedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.1f, 0.0f, 0.7f, -0.6f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(-1.0f, 1.0f, 1.0f, -1.0f))
        val res = signedArray.nonZeroSign()

        it("should return a new DenseNDArray") {
          assertFalse { signedArray === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("sqrt() method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.3162f, 0.4472f, 0.5478f, 0.0f))
        val res = array.sqrt()

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("pow(number) method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.2399f, 0.3687f, 0.4740f, 0.0f))
        val res = array.pow(0.62f)

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("exp() method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(1.105171f, 1.221403f, 1.349859f, 1.0f))
        val res = array.exp()

        it("should return a new DenseNDArray") {
          assertFalse { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("log10() method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.397940f, -0.522879f, -0.301030f, -0.154902f))
        val res = a.log10()

        it("should raise an exception if at least a value is 0.0f") {
          assertFailsWith<IllegalArgumentException> { array.log10() }
        }

        it("should return a new DenseNDArray with a valid array") {
          assertFalse { a === res }
        }

        it("should return the expected values with a valid array") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-06f) }
        }
      }

      context("ln() method") {

        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.916291f, -1.203973f, -0.693147f, -0.356675f))
        val res = a.ln()

        it("should raise an exception if at least a value is 0.0f") {
          assertFailsWith<IllegalArgumentException> { array.ln() }
        }

        it("should return a new DenseNDArray with a valid array") {
          assertFalse { a === res }
        }

        it("should return the expected values with a valid array") {
          assertTrue { res.equals(expectedArray, tolerance = 1.0e-06f) }
        }
      }
    }

    context("math methods in-place") {

      val a = DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.3f, 0.5f, 0.7f))
      val b = DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, 0.8f, 0.1f, 0.4f))
      val n = 0.9f

      context("assignSum(number) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(1.0f, 1.1f, 1.2f, 0.9f))
        val res = array.assignSum(n)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignSum(array, number) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(1.3f, 1.2f, 1.4f, 1.6f))
        val res = array.assignSum(a, n)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignSum(array, array) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(1.1f, 1.1f, 0.6f, 1.1f))
        val res = array.assignSum(a, b)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignSumByRows(array) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val matrix = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f),
          floatArrayOf(0.4f, 0.5f, 0.7f, 0.9f)
        ))
        val expectedRes = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.2f, 0.4f, 0.6f, 0.0f),
          floatArrayOf(0.5f, 0.7f, 1.0f, 0.9f)
        ))
        val res = matrix.assignSumByRows(array)

        it("should return the same DenseNDArray") {
          assertTrue { matrix === res }
        }

        it("should assign the expected values") {
          assertTrue { matrix.equals(expectedRes, tolerance = 1.0e-04f) }
        }
      }

      context("assignSumByColumns(array) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f))
        val matrix = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f),
          floatArrayOf(0.4f, 0.5f, 0.7f, 0.9f)
        ))
        val expectedRes = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.2f, 0.3f, 0.4f, 0.1f),
          floatArrayOf(0.6f, 0.7f, 0.9f, 1.1f)
        ))
        val res = matrix.assignSumByColumns(array)

        it("should return the same DenseNDArray") {
          assertTrue { matrix === res }
        }

        it("should assign the expected values") {
          assertTrue { matrix.equals(expectedRes, tolerance = 1.0e-04f) }
        }
      }

      context("assignSum(array) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, 0.5f, 0.8f, 0.7f))
        val res = array.assignSum(a)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignSub(number) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.8f, -0.7f, -0.6f, -0.9f))
        val res = array.assignSub(n)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignSub(array) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.3f, -0.1f, -0.2f, -0.7f))
        val res = array.assignSub(a)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignDot(array, array[1-d]) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val a1 = DenseNDArrayFactory.arrayOf(floatArrayOf(0.28f))
        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.112f),
          floatArrayOf(0.084f),
          floatArrayOf(0.14f),
          floatArrayOf(0.196f)
        ))
        val res = array.assignDot(a, a1)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should throw an error with not compatible shapes") {
          assertFails { array.assignDot(a, b.t) }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignDot(array, array[2-d]) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val v = DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, 0.8f))
        val m = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.7f, 0.5f),
          floatArrayOf(0.3f, 0.2f),
          floatArrayOf(0.3f, 0.5f),
          floatArrayOf(0.7f, 0.5f)
        ))
        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.61f),
          floatArrayOf(0.25f),
          floatArrayOf(0.49f),
          floatArrayOf(0.61f)
        ))
        val res = array.assignDot(m, v)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should throw an error with not compatible shapes") {
          val m2 = DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.7f, 0.5f),
            floatArrayOf(0.3f, 0.2f),
            floatArrayOf(0.3f, 0.5f),
            floatArrayOf(0.7f, 0.5f)
          ))
          assertFails { array.assignDot(a.t, m2) }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignDotLeftMasked(array[1-d], array[2-d], mask) method") {

        val array = DenseNDArrayFactory.emptyArray(Shape(1, 2))
        val a1 = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.7f, 0.3f, 0.6f)
        ))
        val m = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.5f, 0.3f),
          floatArrayOf(1.0f, 0.5f),
          floatArrayOf(0.7f, 0.6f)
        ))
        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.3f, 0.15f)
        ))
        val res = array.assignDotLeftMasked(a1, m, aMask = NDArrayMask(dim1 = intArrayOf(0), dim2 = intArrayOf(1)))

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should throw an error with not compatible shapes") {
          val m2 = DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.7f, 0.5f),
            floatArrayOf(0.3f, 0.2f),
            floatArrayOf(0.3f, 0.5f),
            floatArrayOf(0.7f, 0.5f)
          ))
          assertFails { array.assignDot(a1, m2) }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignDotLeftMasked(array[2-d], array[2-d], mask) method") {

        val array = DenseNDArrayFactory.emptyArray(Shape(3, 2))
        val m1 = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.5f, 0.3f),
          floatArrayOf(1.0f, 0.5f),
          floatArrayOf(0.7f, 0.6f)
        ))
        val m2 = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.2f, 0.9f),
          floatArrayOf(0.5f, 0.6f)
        ))
        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f, 0.45f),
          floatArrayOf(0.25f, 0.3f),
          floatArrayOf(0.0f, 0.0f)
        ))
        val res = array.assignDotLeftMasked(m1, m2, aMask = NDArrayMask(dim1 = intArrayOf(0, 1), dim2 = intArrayOf(0, 1)))

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should throw an error with not compatible shapes") {
          val m3 = DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.7f, 0.5f),
            floatArrayOf(0.3f, 0.2f),
            floatArrayOf(0.3f, 0.5f),
            floatArrayOf(0.7f, 0.5f)
          ))
          assertFails { array.assignDot(m1, m3) }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignDotRightMasked(array[1-d], array[2-d], mask) method") {

        val array = DenseNDArrayFactory.emptyArray(Shape(1, 2))
        val a1 = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.7f, 0.3f, 0.6f)
        ))
        val m = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.5f, 0.3f),
          floatArrayOf(1.0f, 0.5f),
          floatArrayOf(0.7f, 0.6f)
        ))
        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.35f, 0.15f)
        ))
        val mask = NDArrayMask(dim1 = intArrayOf(0, 1), dim2 = intArrayOf(0, 1))
        val res = array.assignDotRightMasked(a1, m, bMask = mask)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should throw an error with not compatible shapes") {
          val m2 = DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.7f, 0.5f),
            floatArrayOf(0.3f, 0.2f),
            floatArrayOf(0.3f, 0.5f),
            floatArrayOf(0.7f, 0.5f)
          ))
          assertFails { array.assignDot(a1, m2) }
        }

        it("should assign the expected values") {
          assertTrue { expectedArray.equals(array, tolerance = 1.0e-04f) }
        }
      }

      context("assignDotRightMasked(array[2-d], array[2-d], mask) method") {

        val array = DenseNDArrayFactory.emptyArray(Shape(3, 2))
        val m1 = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.5f, 0.3f),
          floatArrayOf(1.0f, 0.5f),
          floatArrayOf(0.7f, 0.6f)
        ))
        val m2 = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.2f, 0.9f),
          floatArrayOf(0.5f, 0.6f)
        ))
        val expectedArray = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f, 0.18f),
          floatArrayOf(0.2f, 0.3f),
          floatArrayOf(0.14f, 0.36f)
        ))
        val mask = NDArrayMask(dim1 = intArrayOf(0, 1), dim2 = intArrayOf(0, 1))
        val res = array.assignDotRightMasked(m1, m2, bMask = mask)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should throw an error with not compatible shapes") {
          val m3 = DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.7f, 0.5f),
            floatArrayOf(0.3f, 0.2f),
            floatArrayOf(0.3f, 0.5f),
            floatArrayOf(0.7f, 0.5f)
          ))
          assertFails { array.assignDot(m1, m3) }
        }

        it("should assign the expected values") {
          assertTrue { expectedArray.equals(array, tolerance = 1.0e-04f) }
        }
      }

      context("assignProd(number) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.09f, 0.18f, 0.27f, 0.0f))
        val res = array.assignProd(n)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignProd(array, number) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.36f, 0.27f, 0.45f, 0.63f))
        val res = array.assignProd(a, n)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignProd(array, array) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.28f, 0.24f, 0.05f, 0.28f))
        val res = array.assignProd(a, b)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignProd(array) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.04f, 0.06f, 0.15f, 0.0f))
        val res = array.assignProd(a)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignDiv(number) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1111f, 0.2222f, 0.3333f, 0.0f))
        val res = array.assignDiv(n)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignDiv(array) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.25f, 0.6667f, 0.6f, 0.0f))
        val res = array.assignDiv(a)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values "+a) {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignPow(number) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.2399f, 0.3687f, 0.4740f, 0.0f))
        val res = array.assignPow(0.62f)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignExp(number) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(1.105171f, 1.221403f, 1.349859f, 1.0f))
        val res = array.assignExp()

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-6f) }
        }
      }

      context("assignSqrt(number) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.3162f, 0.4472f, 0.5478f, 0.0f))
        val res = array.assignSqrt()

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("assignLog10() method") {

        val array1 = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val array2 = DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.3f, 0.5f, 0.7f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.397940f, -0.522879f, -0.301030f, -0.154902f))
        val res = array2.assignLog10()

        it("should raise an exception if at least a value is 0.0f") {
          assertFailsWith<IllegalArgumentException> { array1.assignLog10() }
        }

        it("should return the same DenseNDArray with a valid array") {
          assertTrue { array2 === res }
        }

        it("should assign the expected values with a valid array") {
          assertTrue { array2.equals(expectedArray, tolerance = 1.0e-06f) }
        }
      }

      context("assignLn() method") {

        val array1 = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val array2 = DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.3f, 0.5f, 0.7f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(-0.916291f, -1.203973f, -0.693147f, -0.356675f))
        val res = array2.assignLn()

        it("should raise an exception if at least a value is 0.0f") {
          assertFailsWith<IllegalArgumentException> { array1.assignLn() }
        }

        it("should return the same DenseNDArray with a valid array") {
          assertTrue { array2 === res }
        }

        it("should assign the expected values with a valid array") {
          assertTrue { array2.equals(expectedArray, tolerance = 1.0e-06f) }
        }
      }

      context("assignRoundInt(threshold) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val expectedArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 1.0f, 1.0f, 0.0f))
        val res = array.assignRoundInt(threshold = 0.2f)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should assign the expected values") {
          assertTrue { array.equals(expectedArray, tolerance = 1.0e-04f) }
        }
      }

      context("randomize(randomGenerator) method") {

        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))
        val randomGeneratorMock = mock<RandomGenerator>()
        var i = 0
        @Suppress("UNUSED_CHANGED_VALUE")
        whenever(randomGeneratorMock.next()).then { a[i++] } // assign the same values of [a]

        val res = array.randomize(randomGeneratorMock)

        it("should return the same DenseNDArray") {
          assertTrue { array === res }
        }

        it("should return the expected values") {
          assertTrue { res.equals(a) }
        }
      }
    }

    context("other math methods") {

      val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))

      context("sum() method") {

        it("should give the expected sum of its elements") {
          assertTrue { equals(0.6f, array.sum(), tolerance = 1.0e-10f) }
        }
      }

      context("norm() method") {

        it("should return the expected norm") {
          assertTrue { equals(0.6f, array.norm(), tolerance = 1.0e-05f) }
        }
      }

      context("norm2() method") {

        it("should return the expected euclidean norm") {
          assertTrue { equals(0.37417f, array.norm2(), tolerance = 1.0e-05f) }
        }
      }

      context("argMaxIndex() method") {

        it("should have the expected argmax index") {
          assertEquals(2, array.argMaxIndex())
        }

        it("should have the expected argmax index excluding a given index") {
          assertEquals(1, array.argMaxIndex(exceptIndex = 2))
        }

        it("should have the expected argmax index excluding more indices") {
          assertEquals(0, array.argMaxIndex(exceptIndices = setOf(1, 2)))
        }
      }

      context("max() method") {

        it("should have the expected max value") {
          assertEquals(0.3f, array.max())
        }
      }
    }

    context("initialization through an array of 2 float arrays of 4 elements") {

      val array = DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f),
        floatArrayOf(0.5f, 0.6f, 0.7f, 0.8f)
      ))

      context("properties") {

        it("should not be a vector") {
          assertFalse { array.isVector }
        }

        it("should be a matrix") {
          assertTrue { array.isMatrix }
        }

        it("should have the expected length") {
          assertEquals(8, array.length)
        }

        it("should have the expected number of rows") {
          assertEquals(2, array.rows)
        }

        it("should have the expected number of columns") {
          assertEquals(4, array.columns)
        }

        it("should have the expected shape") {
          assertEquals(Shape(2, 4), array.shape)
        }
      }

      context("generic methods") {

        it("should be equal to itself") {
          assertTrue { array.equals(array) }
        }

        it("should be equal to its copy") {
          assertTrue { array.equals(array.copy()) }
        }
      }

      context("getRange() method") {

        it("should fail the vertical vector require") {
          assertFailsWith<Throwable> {
            array.getRange(2, 4)
          }
        }
      }

      context("getRow() method") {

        val row = array.getRow(1)

        it("should return a row vector") {
          assertEquals(1, row.rows)
        }

        it("should return the expected row values") {
          assertTrue { row.equals(DenseNDArrayFactory.arrayOf(listOf(floatArrayOf(0.5f, 0.6f, 0.7f, 0.8f)))) }
        }
      }

      context("getRows() method") {

        val rows = array.getRows()

        it("should return the expected number of rows") {
          assertEquals(2, rows.size)
        }

        it("should return the expected first row") {
          assertTrue {
            rows[0].equals(DenseNDArrayFactory.arrayOf(listOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f))), tolerance = 0.001f)
          }
        }

        it("should return the expected second row") {
          assertTrue {
            rows[1].equals(DenseNDArrayFactory.arrayOf(listOf(floatArrayOf(0.5f, 0.6f, 0.7f, 0.8f))), tolerance = 0.001f)
          }
        }
      }

      context("getColumn() method") {

        val column = array.getColumn(1)

        it("should return a column vector") {
          assertEquals(1, column.columns)
        }

        it("should return the expected column values") {
          assertTrue { column.equals(DenseNDArrayFactory.arrayOf(floatArrayOf(0.2f, 0.6f))) }
        }
      }

      context("getColumns() method") {

        val columns = array.getColumns()

        it("should return the expected number of columns") {
          assertEquals(4, columns.size)
        }

        it("should return the expected first column") {
          assertTrue {
            columns[0].equals(
              DenseNDArrayFactory.arrayOf(listOf(floatArrayOf(0.1f), floatArrayOf(0.5f))),
              tolerance = 0.001f)
          }
        }

        it("should return the expected second column") {
          assertTrue {
            columns[1].equals(
              DenseNDArrayFactory.arrayOf(listOf(floatArrayOf(0.2f), floatArrayOf(0.6f))),
              tolerance = 0.001f)
          }
        }

        it("should return the expected third column") {
          assertTrue {
            columns[2].equals(
              DenseNDArrayFactory.arrayOf(listOf(floatArrayOf(0.3f), floatArrayOf(0.7f))),
              tolerance = 0.001f)
          }
        }

        it("should return the expected fourth column") {
          assertTrue {
            columns[3].equals(
              DenseNDArrayFactory.arrayOf(listOf(floatArrayOf(0.4f), floatArrayOf(0.8f))),
              tolerance = 0.001f)
          }
        }
      }

      context("transpose") {

        val transposedArray = array.t

        it("should give a transposed array with the expected shape") {
          assertEquals(Shape(4, 2), transposedArray.shape)
        }

        it("should give a transposed array with the expected values") {
          assertEquals(transposedArray[2, 1], 0.7f)
        }
      }
    }

    context("initialization through zerosLike()") {

      val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f)).zerosLike()
      val arrayOfZeros = array.zerosLike()

      it("should have the expected length") {
        assertEquals(array.length, arrayOfZeros.length)
      }

      it("should have the expected values") {
        assertTrue { DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)).equals(arrayOfZeros) }
      }
    }

    context("initialization through onesLike()") {

      val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f)).onesLike()
      val arrayOfOnes = array.onesLike()

      it("should have the expected length") {
        assertEquals(array.length, arrayOfOnes.length)
      }

      it("should have the expected values") {
        assertTrue { DenseNDArrayFactory.arrayOf(floatArrayOf(1.0f, 1.0f, 1.0f, 1.0f)).equals(arrayOfOnes) }
      }
    }

    context("converting a DenseNDArray to zeros") {

      val array = DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f),
        floatArrayOf(0.5f, 0.6f, 0.7f, 0.8f)
      ))

      context("zeros() method call") {

        array.zeros()

        it("should return an DenseNDArray filled with zeros") {
          (0 until array.length).forEach { i -> assertEquals(0.0f, array[i]) }
        }
      }
    }

    context("converting a DenseNDArray to ones") {

      val array = DenseNDArrayFactory.arrayOf(listOf(
        floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f),
        floatArrayOf(0.5f, 0.6f, 0.7f, 0.8f)
      ))

      context("ones() method call") {

        array.ones()

        it("should return an DenseNDArray filled with ones") {
          (0 until array.length).forEach { i -> assertEquals(1.0f, array[i]) }
        }
      }
    }

    context("values assignment") {

      context("assignment through another DenseNDArray") {

        val array = DenseNDArrayFactory.emptyArray(Shape(3, 2))
        val arrayToAssign = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f, 0.2f),
          floatArrayOf(0.3f, 0.4f),
          floatArrayOf(0.5f, 0.6f)
        ))

        array.assignValues(arrayToAssign)

        it("should contain the expected assigned values") {
          assertTrue { array.equals(arrayToAssign) }
        }
      }

      context("assignment through a number") {

        val array = DenseNDArrayFactory.emptyArray(Shape(3, 2))

        array.assignValues(0.6f)

        it("should contain the expected assigned values") {
          (0 until array.length).forEach { i -> assertEquals(0.6f, array[i]) }
        }
      }
    }

    context("getters") {

      context("a vertical vector") {
        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))

        it("should get the correct item") {
          assertEquals(array[2], 0.3f)
        }
      }

      context("a horizontal vector") {
        val array = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f),
          floatArrayOf(0.2f),
          floatArrayOf(0.3f),
          floatArrayOf(0.0f)
        ))

        it("should get the correct item") {
          assertEquals(array[2], 0.3f)
        }
      }

      context("a matrix") {
        val array = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f, 0.2f, 0.3f),
          floatArrayOf(0.4f, 0.5f, 0.6f)
        ))

        it("should get the correct item") {
          assertEquals(array[1, 2], 0.6f)
        }
      }
    }

    context("setters") {

      context("a vertical vector") {
        val array = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.0f))

        array[2] = 0.7f

        it("should set the correct item") {
          assertEquals(array[2], 0.7f)
        }
      }

      context("a horizontal vector") {
        val array = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f),
          floatArrayOf(0.2f),
          floatArrayOf(0.3f),
          floatArrayOf(0.0f)
        ))

        array[2] = 0.7f

        it("should get the correct item") {
          assertEquals(array[2], 0.7f)
        }
      }

      context("a matrix") {
        val array = DenseNDArrayFactory.arrayOf(listOf(
          floatArrayOf(0.1f, 0.2f, 0.3f),
          floatArrayOf(0.4f, 0.5f, 0.6f)
        ))

        array[1, 2] = 0.7f

        it("should get the correct item") {
          assertEquals(array[1, 2], 0.7f)
        }
      }
    }

    context("single horizontal concatenation") {

      val array1 = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f))
      val array2 = DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.5f, 0.6f))
      val concatenatedArray = array1.concatH(array2)

      it("should have the expected shape") {
        assertEquals(Shape(3, 2), concatenatedArray.shape)
      }

      it("should have the expected values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.1f, 0.4f),
            floatArrayOf(0.2f, 0.5f),
            floatArrayOf(0.3f, 0.6f))
          ).equals(concatenatedArray)
        }
      }
    }

    context("single vertical concatenation") {

      val array1 = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f))
      val array2 = DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.5f, 0.6f))
      val concatenatedArray = array1.concatV(array2)

      it("should have the expected length") {
        assertEquals(6, concatenatedArray.length)
      }

      it("should have the expected values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f))
            .equals(concatenatedArray)
        }
      }
    }

    context("multiple vertical concatenation") {

      val concatenatedArray = concatVectorsV(
        DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f)),
        DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.5f, 0.6f)),
        DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, 0.8f, 0.9f))
      )

      it("should have the expected length") {
        assertEquals(9, concatenatedArray.length)
      }

      it("should have the expected values") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f))
            .equals(concatenatedArray)
        }
      }
    }

    context("single vertical split") {

      val array1 = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f))

      val splitArray: List<DenseNDArray> = array1.splitV(2)

      it("should have the expected length") {
        assertEquals(2, splitArray.size)
      }

      it("should have the expected values") {
        assertEquals(
          listOf(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f)),
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f, 0.4f))
          ),
          splitArray
        )
      }
    }

    context("single vertical split multiple range size") {

      val array1 = DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f))

      val splitArray: List<DenseNDArray> = array1.splitV(2, 1, 1)

      it("should have the expected length") {
        assertEquals(3, splitArray.size)
      }

      it("should have the expected values") {
        assertEquals(
          listOf(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.1f, 0.2f)),
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.3f)),
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f))
          ),
          splitArray
        )
      }
    }
  }
})
