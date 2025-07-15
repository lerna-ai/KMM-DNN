/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray.sparse

import com.kotlinnlp.simplednn.simplemath.ndarray.Indices
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.SparseEntry

/**
 *
 */
object SparseNDArrayFactory : NDArrayFactory<SparseNDArray> {

  /**
   * Private val used to serialize the class (needed by Serializable).
   */
  @Suppress("unused")
  private const val serialVersionUID: Long = 1L

  /**
   * @param shape shape
   *
   * @return a new empty [SparseNDArray]
   */
  override fun emptyArray(shape: Shape) = SparseNDArray(
    shape = shape,
    rows = intArrayOf(),
    columns = intArrayOf(),
    values = doubleArrayOf()
  )

  /**
   * Build a new [SparseNDArray] filled with zeros.
   *
   * @param shape shape
   *
   * @return a new [SparseNDArray]
   */
  override fun zeros(shape: Shape) = SparseNDArray(shape = shape)

  /**
   * Build a new diagonal [SparseNDArray] filled with ones.
   *
   * @param size the number of rows and columns
   *
   * @return a new [SparseNDArray]
   */
  override fun eye(size: Int): SparseNDArray = this.arrayOf(
    activeIndicesValues = Array(size) { i -> SparseEntry(Indices(i, i), 1.0) },
    shape = Shape(size, size))

  /**
   * Build a new [SparseNDArray] filled with a constant value.
   *
   * @param shape shape
   * @param value the init value
   *
   * @return a new [SparseNDArray]
   */
  override fun fill(shape: Shape, value: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Build a new [SparseNDArray] filled with zeros but one with 1.0.
   *
   * @param length the length of the array
   * @param oneAt the index of the one element
   *
   * @return a oneHotEncoder [SparseNDArray]
   */
  override fun oneHotEncoder(length: Int, oneAt: Int): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Build a new [SparseNDArray] filled with random values uniformly distributed in range [[from], [to]].
   *
   * @param shape shape
   * @param from inclusive lower bound of random values range
   * @param to inclusive upper bound of random values range
   *
   * @return a new [SparseNDArray] filled with random values
   */
  override fun random(shape: Shape, from: Double, to: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  fun arrayOf(activeIndicesValues: Array<SparseEntry>, shape: Shape): SparseNDArray {

    val values = mutableListOf<Double>()
    val rows = mutableListOf<Int>()
    val columns = mutableListOf<Int>()

    for ((indices, value) in activeIndicesValues.sortedWith(Comparator { (aIndices), (bIndices) ->
      if (aIndices.second != bIndices.second) {
        aIndices.second - bIndices.second
      } else {
        aIndices.first - bIndices.first
      }
    })) {
      require(indices.first < shape.dim1 && indices.second < shape.dim2) {
        "Indices out of bounds (%d, %d)".format(indices.first, indices.second)
      }

      if (value != 0.0) {
        values.add(value)
        rows.add(indices.first)
        columns.add(indices.second)
      }
    }

    return SparseNDArray(
      shape = shape,
      rows = rows.toIntArray(),
      columns = columns.toIntArray(),
      values = values.toDoubleArray())
  }
}
