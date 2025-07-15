/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray.dense

import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.api.rand
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.data.D1Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.times

/**
 *
 */
object DenseNDArrayFactory : NDArrayFactory<DenseNDArray> {

  /**
   * Private val used to serialize the class (needed by Serializable).
   */
  @Suppress("unused")
  private const val serialVersionUID: Long = 1L

  /**
   * @param shape shape
   *
   * @return a new empty [DenseNDArray]
   */
  override fun emptyArray(shape: Shape): DenseNDArray =
    DenseNDArray(mk.zeros<Double>(shape.dim1, shape.dim2))

  /**
   * Build a new [DenseNDArray] filled with zeros.
   *
   * @param shape shape
   *
   * @return a new [DenseNDArray]
   */
  override fun zeros(shape: Shape): DenseNDArray =
    this.emptyArray(shape)

  /**
   * Build a new [DenseNDArray] filled with ones.
   *
   * @param shape shape
   *
   * @return a new [DenseNDArray]
   */
  override fun ones(shape: Shape): DenseNDArray =
    DenseNDArray(mk.ones<Double>(shape.dim1, shape.dim2))

  /**
   * Build a new diagonal [DenseNDArray] filled with ones.
   *
   * @param size the number of rows and columns
   *
   * @return a new [DenseNDArray]
   */
  override fun eye(size: Int): DenseNDArray =
    this.zeros(Shape(size, size)).apply {
      (0 until rows).forEach { i -> set(i, i, 1.0) }
    }

  fun eye(size: Int, a: D1Array<Double>): DenseNDArray =
    this.zeros(Shape(size, size)).apply {
      (0 until rows).forEach { i -> set(i, i, a[i]) }
    }

  /**
   * @param shape shape
   * @param value the init value
   *
   * @return a new [DenseNDArray] filled with the given value
   */
  override fun fill(shape: Shape, value: Double): DenseNDArray {
    var temp = mk.zeros<Double>(shape.dim1, shape.dim2)
    temp+=value
    return DenseNDArray(temp)
  }

  /**
   * Build a new [DenseNDArray] filled with zeros but one with 1.0.
   *
   * @param length the length of the array
   * @param oneAt the index of the one element
   *
   * @return a oneHotEncoder [DenseNDArray]
   */
  override fun oneHotEncoder(length: Int, oneAt: Int): DenseNDArray {
    require(oneAt in 0 until length)

    val array = this.emptyArray(Shape(length))

    array[oneAt] = 1.0

    return array
  }

  /**
   * Build a new [DenseNDArray] filled with random values uniformly distributed in range [[from], [to]].
   *
   * @param shape shape
   * @param from inclusive lower bound of random values range
   * @param to inclusive upper bound of random values range
   *
   * @return a new [DenseNDArray] filled with random values
   */
  override fun random(shape: Shape, from: Double, to: Double): DenseNDArray {

    var m = mk.rand<Double>(shape.dim1, shape.dim2)
    val rangeSize = to - from

    if (rangeSize != 1.0) {
      m*=(rangeSize)
    }

    if (from != 0.0) {
      m+=(from)
    }

    return DenseNDArray(m)
  }

  /**
   * @param values an array of double numbers
   *
   * @return a new column vector filled with the given values
   */
  fun arrayOf(values: DoubleArray): DenseNDArray {

    var m = mk.zeros<Double>(values.size, 1)

    values.indices.forEach { i -> m[i, 0] = values[i] }

    return DenseNDArray(m)
  }

  /**
   * @param rows rows as arrays of double numbers
   *
   * @return a new dense array filled by rows with the given values
   */
  fun arrayOf(rows: List<DoubleArray>): DenseNDArray {

    val dim1 = rows.size
    val dim2 = if (rows.isNotEmpty()) rows[0].size else 0
    var m = mk.zeros<Double>(dim1, dim2)

    (0 until dim1 * dim2).forEach { linearIndex ->
      // linear indexing: loop rows before, column by column
      val row = linearIndex % dim1
      val column = linearIndex / dim1
      m[row, column] = rows[row][column]
    }

    return DenseNDArray(m)
  }

  /**
   * @param rows rows as dense arrays
   *
   * @return a new dense array filled by rows with the given values
   */
  fun fromRows(rows: List<DenseNDArray>): DenseNDArray {

    val dim1 = rows.size
    val dim2 = if (rows.isNotEmpty()) rows[0].length else 0
    var m = mk.zeros<Double>(dim1, dim2)

    require(rows.all { it.length == dim2 }) { "All the rows must have the same length. "}

    (0 until dim1 * dim2).forEach { linearIndex ->
      // linear indexing: loop rows before, column by column
      val row = linearIndex % dim1
      val column = linearIndex / dim1
      m[row, column] = rows[row][column]
    }

    return DenseNDArray(m)
  }

  /**
   * @param columns columns as dense arrays
   *
   * @return a new dense array filled by columns with the given values
   */
  fun fromColumns(columns: List<DenseNDArray>): DenseNDArray {

    val dim1 = if (columns.isNotEmpty()) columns[0].length else 0
    val dim2 = columns.size
    var m = mk.zeros<Double>(dim1, dim2)

    require(columns.all { it.length == dim1 }) { "All the columns must have the same length. "}

    (0 until dim1 * dim2).forEach { linearIndex ->
      // linear indexing: loop rows before, column by column
      val row = linearIndex % dim1
      val column = linearIndex / dim1
      m[row, column] = columns[column][row]
    }

    return DenseNDArray(m)
  }

  /**
   * Build a [DenseNDArray] that contains the same values of a given generic NDArray.
   *
   * @param array a generic NDArray
   *
   * @return a new Dense NDArray
   */
  fun fromNDArray(array: NDArray<*>): DenseNDArray = when (array) {

    is DenseNDArray -> array.copy()

    is SparseNDArray -> {
      val ret: DenseNDArray = zeros(array.shape)
      array.forEach { (indices, value) -> ret[indices.first, indices.second] = value }
      ret
    }

    is SparseBinaryNDArray -> {
      val ret: DenseNDArray = zeros(array.shape)
      array.forEach { (i, j) -> ret[i, j] = 1.0 }
      ret
    }

    else -> throw RuntimeException("Invalid NDArray type.")
  }

  /**
   * Creates an array with a single element.
   *
   * @param value the value
   *
   * @return a new [DenseNDArray] filled with the given value
   */
  fun scalarOf(value: Double) = arrayOf(doubleArrayOf(value))
}
