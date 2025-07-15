/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray.dense

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.ndarray.*
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import org.jetbrains.kotlinx.multik.api.Multik.linalg
import org.jetbrains.kotlinx.multik.api.d2arrayIndices
import org.jetbrains.kotlinx.multik.api.linalg.Norm
import org.jetbrains.kotlinx.multik.api.linalg.norm
import org.jetbrains.kotlinx.multik.api.linalg.svd
import org.jetbrains.kotlinx.multik.api.linalg.dot
import org.jetbrains.kotlinx.multik.api.math.exp
import org.jetbrains.kotlinx.multik.api.math.log
import org.jetbrains.kotlinx.multik.ndarray.data.D2Array
import org.jetbrains.kotlinx.multik.ndarray.data.get
import org.jetbrains.kotlinx.multik.ndarray.data.set
import org.jetbrains.kotlinx.multik.ndarray.operations.toDoubleArray
import org.jetbrains.kotlinx.multik.api.stat.abs
import org.jetbrains.kotlinx.multik.api.mk
import org.jetbrains.kotlinx.multik.api.ones
import org.jetbrains.kotlinx.multik.api.zeros
import org.jetbrains.kotlinx.multik.ndarray.operations.average
import org.jetbrains.kotlinx.multik.ndarray.operations.plus
import org.jetbrains.kotlinx.multik.ndarray.operations.sum
import org.jetbrains.kotlinx.multik.ndarray.operations.minus
import org.jetbrains.kotlinx.multik.ndarray.operations.times
import org.jetbrains.kotlinx.multik.ndarray.operations.div
import org.jetbrains.kotlinx.multik.ndarray.operations.max
import org.jetbrains.kotlinx.multik.ndarray.operations.min
import kotlin.math.*

/**
 * [NDArray] with dense values (implemented using JBlas)
 */
//@kotlinx.serialization.Serializable

class DenseNDArray(private var storage: D2Array<Double>) : NDArray<DenseNDArray> {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L
  }

  /**
   *
   */
  override val factory = DenseNDArrayFactory

  /**
   *
   */

  override val rows: Int
    get() = this.storage.shape[0]

  /**
   *
   */
  override val columns: Int
    get() = this.storage.shape[1]

  /**
   * Whether the array is a row or a column vector
   */
  override val isVector: Boolean
    get() = this.rows == 1 || this.columns == 1

  /**
   *
   */
  override val isOneHotEncoder: Boolean get() {

    var isOneHot = false

    if (this.isVector) {
      (0 until this.length)
        .asSequence()
        .filter { this[it] != 0.0 }
        .forEach {
          if (this[it] == 1.0 && !isOneHot)
            isOneHot = true
          else
            return false
        }
    }

    return isOneHot
  }


  /**
   *
   */
  override val length: Int get() = this.storage.shape[0]*this.storage.shape[1]

  /**
   *
   */
  override val lastIndex: Int = this.length - 1

  /**
   *
   */
  override val shape: Shape get() = Shape(this.rows, this.columns)

  /**
   *
   */
  override val t: DenseNDArray get() = DenseNDArray(this.storage.transpose())

  /**
   *
   */
  override operator fun get(i: Int): Double = this.storage[i%this.rows,i/this.rows]

  /**
   *
   */
  override operator fun get(i: Int, j: Int): Double = this.storage[i, j]

  /**
   *
   */
  override operator fun set(i: Int, value: Number) {
    this.storage[i%this.rows, i/this.rows] = value.toDouble()
  }

  /**
   *
   */
  override operator fun set(i: Int, j: Int, value: Number) {
    this.storage[i, j] = value.toDouble()
  }

  /**
   * Get the i-th row
   *
   * @param i the index of the row to be returned
   *
   * @return the selected row as a new DenseNDArray
   */
  override fun getRow(i: Int): DenseNDArray {
    val values = this.storage[i]
    return DenseNDArrayFactory.arrayOf(listOf(values.toDoubleArray()))
  }

  /**
   * Get the i-th column
   *
   * @param i the index of the column to be returned
   *
   * @return the selected column as a new DenseNDArray
   */
  override fun getColumn(i: Int): DenseNDArray {
    val values = this.storage[0..this.rows-1,i]
    return DenseNDArrayFactory.arrayOf(values.toDoubleArray())
  }

  /**
   * Get a one-dimensional DenseNDArray sub-vector of a vertical vector.
   *
   * @param a the start index of the range (inclusive)
   * @param b the end index of the range (exclusive)
   *
   * @return the sub-array
   */
  override fun getRange(a: Int, b: Int): DenseNDArray {
    require(this.shape.dim2 == 1)
    val values = this.storage[a..b-1,0]
    return DenseNDArrayFactory.arrayOf(values.toDoubleArray())
  }

  /**
   * Returns a list containing the results of applying the given [transform] function
   * to each element in the original collection.
   */
  inline fun map(transform: (Double) -> Double): DenseNDArray {
    return this.mapTo(DenseNDArrayFactory.zeros(this.shape), transform)
  }

  /**
   * Applies the given [transform] function to each element of the original collection
   * and appends the results to the given [destination].
   */
  inline fun mapTo(destination: DenseNDArray, transform: (Double) -> Double): DenseNDArray {

    (0 until destination.length).forEach { i -> destination[i] = transform(this[i]) }

    return destination
  }

  /**
   *
   */
  override fun zeros(): DenseNDArray {
    this.storage=mk.zeros(this.rows, this.columns)
    return this
  }

  /**
   * Fill the array with ones.
   */
  fun ones(): DenseNDArray {
    this.storage=mk.ones(this.rows, this.columns)
    return this
  }

  /**
   * @return a new [DenseNDArray] with the same shape of this, filled with zeros.
   */
  override fun zerosLike(): DenseNDArray = DenseNDArray(mk.zeros(this.rows, this.columns))

  /**
   * @return a new [DenseNDArray] with the same shape of this, filled with ones.
   */
  fun onesLike(): DenseNDArray = DenseNDArray(mk.ones(this.rows, this.columns))

  /**
   *
   */
  override fun copy(): DenseNDArray = DenseNDArray(this.storage.deepCopy())

  /**
   *
   */
  override fun assignValues(n: Double): DenseNDArray {
    this.storage = mk.zeros<Double>(this.rows, this.columns).plus(n)
    return this
  }

  /**
   * Assign the values of [a] to this [DenseNDArray] (it works also among rows and columns vectors).
   *
   * @param a a generic [NDArray]
   *
   * @return this [DenseNDArray]
   */
  override fun assignValues(a: NDArray<*>): DenseNDArray {

    require(this.shape == a.shape ||
            (this.isVector && a.isVector && this.length == a.length))

    when(a) {
      is DenseNDArray -> this.assignValues(a)
      is SparseNDArray -> this.assignValues(a)
      is SparseBinaryNDArray -> TODO("not implemented")
    }

    return this
  }

  /**
   * Assign the values of [a] to this [DenseNDArray] (it works also among rows and columns vectors).
   *
   * @param a a [DenseNDArray]
   */
  private fun assignValues(a: DenseNDArray) {
    (0 until this.rows * this.columns).forEach { linearIndex ->
      // linear indexing: loop rows before, column by column
      val rowa = linearIndex % a.rows
      val columna = linearIndex / a.rows % a.columns
      val rowt = linearIndex % this.rows
      val columnt = linearIndex / this.rows
      this.storage[rowt, columnt] = a.storage[rowa, columna]
    }
  }

  /**
   * Assign the values of [a] to this [DenseNDArray] (it works also among rows and columns vectors).
   *
   * @param a a [SparseNDArray]
   */
  private fun assignValues(a: SparseNDArray) {

    this.zeros()

    a.values.indices.forEach { k ->
      this[a.rowIndices[k], a.colIndices[k]] = a.values[k]
    }
  }

  /**
   *
   */
  override fun assignValues(a: NDArray<*>, mask: NDArrayMask): DenseNDArray {

    require(a.shape == this.shape) { "Arrays with different size" }

    when(a) {
      is DenseNDArray -> this.assignValues(a, mask)
      is SparseNDArray -> this.assignValues(a, mask)
      is SparseBinaryNDArray -> TODO("not implemented")
    }

    return this
  }

  /**
   *
   */
  private fun assignValues(a: DenseNDArray, mask: NDArrayMask): DenseNDArray {

    for (index in 0 until mask.size) {
      val i = mask.dim1[index]
      val j = mask.dim2[index]
      this.storage[i, j] = a[i, j]
    }

    return this
  }

  /**
   *
   */
  private fun assignValues(a: SparseNDArray, mask: NDArrayMask): DenseNDArray {

    require(a.values.size == mask.size) { "Mask has a different number of active values respect of a" }

    for (index in 0 until mask.size) {
      val i = mask.dim1[index]
      val j = mask.dim2[index]
      this.storage[i, j] = a.values[index]
    }

    return this
  }

  /**
   *
   */
  override fun sum(): Double = this.storage.sum()

  /**
   *
   */
  override fun sum(n: Double): DenseNDArray = DenseNDArray(this.storage.plus(n))

  /**
   *
   */
  fun sum(a: NDArray<*>): DenseNDArray = when (a) {
    is DenseNDArray -> this.sum(a)
    is SparseNDArray -> this.copy().assignSum(a)
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  override fun sum(a: DenseNDArray): DenseNDArray {
    val sumValues = DenseNDArrayFactory.emptyArray(this.shape)
    (0 until this.rows * this.columns).forEach { linearIndex ->
      // linear indexing: loop rows before, column by column
      val rowa = linearIndex % a.rows
      val columna = linearIndex / a.rows % a.columns
      val rowt = linearIndex % this.rows
      val columnt = linearIndex / this.rows
      sumValues.storage[rowt, columnt] = this.storage[rowt, columnt] + a.storage[rowa, columna]
    }
    return sumValues
  }
  /**
   *
   */
  override fun sumByRows(a: NDArray<*>): DenseNDArray = when (a) {
    is DenseNDArray -> this.sumByRows(a)
    is SparseNDArray -> TODO("not implemented")
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  private fun sumByRows(a: DenseNDArray): DenseNDArray {

    return if (a.shape == this.shape)
      DenseNDArray(this.storage.plus(a.storage))

    else
      DenseNDArray(mk.d2arrayIndices(
        this.storage.shape[0],
        this.storage.shape[1])
      { i,j -> this.storage[i, j] + a.storage[j, 0] } // linear indexing
      )

  }
  /**
   *
   */
  override fun sumByColumns(a: NDArray<*>): DenseNDArray = when (a) {
    is DenseNDArray -> this.sumByColumns(a)
    is SparseNDArray -> TODO("not implemented")
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  private fun sumByColumns(a: DenseNDArray): DenseNDArray {

    return if (a.shape == this.shape)
      DenseNDArray(this.storage.plus(a.storage))

    else
      DenseNDArray(mk.d2arrayIndices(
        this.storage.shape[0],
        this.storage.shape[1])
      { i,j -> this.storage[i, j] + a.storage[i, 0] } // linear indexing
      )

  }
  /**
   *
   */
  override fun assignSum(n: Double): DenseNDArray {
    this.storage = this.storage.plus(n)
    return this
  }

  fun assignSum(a: DenseNDArray): DenseNDArray {

    val sumValues = DenseNDArrayFactory.emptyArray(this.shape)
    (0 until this.rows * this.columns).forEach { linearIndex ->
      // linear indexing: loop rows before, column by column
      val rowa = linearIndex % a.rows
      val columna = linearIndex / a.rows % a.columns
      val rowt = linearIndex % this.rows
      val columnt = linearIndex / this.rows
      sumValues.storage[rowt, columnt] = this.storage[rowt, columnt] + a.storage[rowa, columna]
    }
    this.storage = sumValues.storage
    return this
  }
  /**
   * Assign a to this DenseNDArray (it works also among rows and columns vectors)
   */
  override fun assignSum(a: NDArray<*>): DenseNDArray {

    when(a) {
      is DenseNDArray -> this.assignSum(a)
      is SparseNDArray -> this.assignSum(a)
      is SparseBinaryNDArray -> TODO("not implemented")
    }

    return this
  }

  /**
   *
   */
  private fun assignSum(a: SparseNDArray): DenseNDArray {

    a.values.indices.forEach { i ->
      this.storage[a.rowIndices[i], a.colIndices[i]] = this.storage[a.rowIndices[i], a.colIndices[i]] + a.values[i]
    }

    return this
  }

  /**
   *
   */
  override fun assignSum(a: DenseNDArray, n: Double): DenseNDArray {
    this.storage = a.storage.plus(n)
    return this
  }

  /**
   * Assign a + b to this DenseNDArray (it works also among rows and columns vectors)
   */
  override fun assignSum(a: DenseNDArray, b: DenseNDArray): DenseNDArray {
    this.storage =a.storage.plus(b.storage)
    return this
  }

  /**
   *
   */
  fun assignSumByRows(a: DenseNDArray): DenseNDArray = this.apply {
    if (a.shape == this.shape)
      this.storage = this.storage.plus(a.storage)

    else{
      this.storage = mk.d2arrayIndices(
        this.storage.shape[0],
        this.storage.shape[1])
      { i,j -> this.storage[i, j] + a.storage[j, 0] } // linear indexing
    }
    return this
  }

  /**
   *
   */
  fun assignSumByColumns(a: DenseNDArray): DenseNDArray = this.apply {
    if (a.shape == this.shape)
      this.storage = this.storage.plus(a.storage)

    else{
      this.storage = mk.d2arrayIndices(
        this.storage.shape[0],
        this.storage.shape[1])
      { i,j -> this.storage[i, j] + a.storage[i, 0] } // linear indexing
    }
    return this
  }

  /**
   *
   */
  override fun sub(n: Double): DenseNDArray = DenseNDArray(this.storage.minus(n))

  /**
   *
   */
  override fun sub(a: NDArray<*>): DenseNDArray = when(a) {
    is DenseNDArray -> this.sub(a)
    is SparseNDArray -> TODO("not implemented")
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  private fun sub(a: DenseNDArray): DenseNDArray = DenseNDArray(this.storage.minus(a.storage))

  /**
   * In-place subtraction by number
   */
  override fun assignSub(n: Double): DenseNDArray {
    this.storage = this.storage.minus(n)
    return this
  }

  fun assignSub(a: DenseNDArray): DenseNDArray {

    this.storage -= a.storage
    return this
  }
  /**
   *
   */
  override fun assignSub(a: NDArray<*>): DenseNDArray {

    when(a) {
      is DenseNDArray -> this.assignSub(a)
      is SparseNDArray -> this.assignSub(a)
      is SparseBinaryNDArray -> TODO("not implemented")
    }

    return this
  }

  /**
   *
   */
  private fun assignSub(a: SparseNDArray): DenseNDArray {

    require(a.shape == this.shape) { "Arrays with different size" }

    a.values.indices.forEach { k ->
      this[a.rowIndices[k], a.colIndices[k]] -= a.values[k]
    }

    return this
  }

  /**
   *
   */
  override fun reverseSub(n: Double): DenseNDArray{
//    val temp = DenseNDArrayFactory.emptyArray(this.shape)
//    for (i in 0 until this.rows)
//      for (j in 0 until this.columns)
//        temp.storage[i, j] = n - this.storage[i,j]
//    return temp

    return DenseNDArray(n - this.storage)
  }

  /**
   * Dot product between this [DenseNDArray] and a [DenseNDArray] masked by a [mask].
   *
   * @param a the array by which is calculated the dot product
   * @param mask the mask applied to a
   *
   * @return the result of the dot product
   */
  fun dotRightMasked(a: DenseNDArray, mask: NDArrayMask): DenseNDArray {

    require(this.columns == a.rows)

    val ret = DenseNDArrayFactory.zeros(shape = Shape(this.rows, a.columns))

    (0 until this.rows).forEach { i ->
      mask.forEach { (k, j) ->
        ret[i, j] += this[i, k] * a[k, j]
      }
    }

    return ret
  }

  /**
   * Dot product between this [DenseNDArray] masked by a [mask] and a [DenseNDArray] .
   *
   * @param a the array by which is calculated the dot product
   * @param mask the mask applied to this array
   *
   * @return the result of the dot product
   */
  fun dotLeftMasked(a: DenseNDArray, mask: NDArrayMask): DenseNDArray {

    require(this.columns == a.rows)

    val ret = DenseNDArrayFactory.zeros(shape = Shape(this.rows, a.columns))

    (0 until a.columns).forEach { j ->
      mask.forEach { (i, k) ->
        ret[i, j] += this[i, k] * a[k, j]
      }
    }

    return ret
  }

  /**
   *
   */
  override fun dot(a: NDArray<*>): DenseNDArray = when(a) {
    is DenseNDArray -> this.dot(a)
    is SparseNDArray -> TODO("not implemented")
    is SparseBinaryNDArray -> this.dot(a)
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  private fun dot(a: DenseNDArray): DenseNDArray = DenseNDArray(this.storage.dot(a.storage))

  /**
   *
   */
  private fun dot(a: SparseBinaryNDArray): DenseNDArray {
    require(this.columns == a.rows)

    val res = DenseNDArrayFactory.zeros(shape = Shape(this.rows, a.columns))

    when {
      a.rows == 1 -> // Column vector (dot) row vector
        for (j in a.activeIndicesByColumn.keys) {
          for (i in 0 until this.rows) {
            res.storage[i, j] = this[i]
          }
        }
      a.columns == 1 -> // n-dim array (dot) column vector
        for (j in a.activeIndicesByRow.keys) {
          for (i in 0 until this.rows) {
            res.storage[i, j] = this[i]
          }
        }
      else -> // n-dim array (dot) n-dim array
        TODO("not implemented")
    }

    return res
  }

  /**
   *
   */
  override fun assignDot(a: DenseNDArray, b: DenseNDArray): DenseNDArray {
    require(a.rows == this.rows && b.columns == this.columns)
    this.storage = a.storage.dot(b.storage)
    return this
  }

  /**
   *
   */
  override fun assignDot(a: DenseNDArray, b: NDArray<*>): DenseNDArray {

    when(b) {
      is DenseNDArray -> this.assignDot(a, b)
      is SparseNDArray -> TODO("not implemented")
      is SparseBinaryNDArray -> this.assignDot(a, b)
    }

    return this
  }

  /**
   *
   */
  private fun assignDot(a: DenseNDArray, b: SparseBinaryNDArray): DenseNDArray {
    require(a.rows == this.rows && b.columns == this.columns && a.columns == b.rows)

    this.zeros()

    when {
      b.rows == 1 -> // Column vector (dot) row vector
        for (j in b.activeIndicesByColumn.keys) {
          for (i in 0 until a.rows) {
            this.storage[i, j] = a[i]
          }
        }
      b.columns == 1 -> // n-dim array (dot) column vector
        //for (j in b.activeIndicesByRow.keys) {
        for (i in 0 until a.rows) {
          this.storage[i,0] = b.activeIndicesByRow.keys.sumOf { a[i, it] }
        }
      // }
      else -> // n-dim array (dot) n-dim array
        TODO("not implemented")
    }

    return this
  }

  /**
   * In-place dot product between the arrays [a] and [b] with a mask applied to [a], assigning the result to this array.
   *
   * @param a an array
   * @param b an array
   * @param aMask the mask applied to [a]
   *
   * @return this array
   */
  fun assignDotLeftMasked(a: DenseNDArray, b: DenseNDArray, aMask: NDArrayMask): DenseNDArray {

    require(a.rows == this.rows && b.columns == this.columns && a.columns == b.rows)

    this.zeros()

    for (j in 0 until b.shape.dim2) {
      for ((i, k) in aMask) {
        this[i, j] += a[i, k] * b[k, j]
      }
    }

    return this
  }

  /**
   * In-place dot product between the arrays [a] and [b] with a mask applied to [b], assigning the result to this array.
   *
   * @param a an array
   * @param b an array
   * @param bMask the mask applied to [b]
   *
   * @return this array
   */
  fun assignDotRightMasked(a: DenseNDArray, b: DenseNDArray, bMask: NDArrayMask): DenseNDArray {

    require(a.rows == this.rows && b.columns == this.columns && a.columns == b.rows)

    this.zeros()

    for (i in 0 until a.shape.dim1) {
      for ((k, j) in bMask) {
        this[i, j] += a[i, k] * b[k, j]
      }
    }

    return this
  }

  /**
   *
   */
  override fun prod(n: Double): DenseNDArray = DenseNDArray(this.storage.times(n))

  /**
   *
   */
  override fun prod(a: NDArray<*>): DenseNDArray = when(a) {
    is DenseNDArray -> this.prod(a)
    is SparseNDArray -> TODO("not implemented")
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   * Product by a [DenseNDArray] with the same shape or a compatible column vector (each column is multiplied
   * by the given vector). It works also against a row and a column vector.
   *
   * @param a the [DenseNDArray] by which this [DenseNDArray] will be multiplied
   *
   * @return a new [DenseNDArray] containing the product between this [DenseNDArray] and [a]
   */
  private fun prod(a: DenseNDArray): DenseNDArray {
    require(a.shape == this.shape ||
            (a.columns == 1 && a.rows == this.rows) ||
            (a.isVector && this.isVector && a.length == this.length)) { "Arrays with not compatible size" }

    if (a.shape == this.shape)
      return DenseNDArray(this.storage.times(a.storage))

    else {

      val prodValues = DenseNDArrayFactory.emptyArray(this.shape)
      (0 until this.rows * this.columns).forEach { linearIndex ->
        // linear indexing: loop rows before, column by column
        val rowa = linearIndex % a.rows
        val columna = (linearIndex / a.rows) % a.columns
        val rowt = linearIndex % this.rows
        val columnt = linearIndex / this.rows
        prodValues.storage[rowt, columnt] = this.storage[rowt, columnt] * a.storage[rowa, columna]
      }

      return prodValues

    }
  }

  /**
   *
   */
  override fun prod(n: Double, mask: NDArrayMask): SparseNDArray {

    val values = DoubleArray(size = mask.size, init = { this.storage[mask.dim1[it], mask.dim2[it]] * n })

    return SparseNDArray(shape = this.shape, values = values, rows = mask.dim1, columns = mask.dim2)
  }

  /**
   *
   */
  override fun assignProd(n: Double): DenseNDArray {
    this.storage = this.storage.times(n)
    return this
  }

  /**
   *
   */
  override fun assignProd(n: Double, mask: NDArrayMask): DenseNDArray {

    for (index in 0 until mask.size) {
      this.storage[mask.dim1[index], mask.dim2[index]] = this.storage[mask.dim1[index], mask.dim2[index]] * n
    }

    return this
  }

  /**
   *
   */
  override fun assignProd(a: DenseNDArray, n: Double): DenseNDArray {
    this.storage = a.storage.times(n)
    return this
  }

  /**
   *
   */
  override fun assignProd(a: DenseNDArray, b: DenseNDArray): DenseNDArray {
    this.storage = a.storage.times(b.storage)
    return this
  }

  /**
   *
   */
  override fun assignProd(a: NDArray<*>): DenseNDArray = when(a) {
    is DenseNDArray -> this.assignProd(a)
    is SparseNDArray -> this.assignProd(a)
    is SparseBinaryNDArray -> this.assignProd(a)
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  private fun assignProd(a: DenseNDArray): DenseNDArray {
    require(a.shape == this.shape ||
            (a.columns == 1 && a.rows == this.rows) ||
            (a.isVector && this.isVector && a.length == this.length)) { "Arrays with not compatible size" }

    if (a.shape == this.shape)
      this.storage = DenseNDArray(this.storage.times(a.storage)).storage

    else {

      val prodValues = DenseNDArrayFactory.emptyArray(this.shape)
      (0 until this.rows * this.columns).forEach { linearIndex ->
        // linear indexing: loop rows before, column by column
        val rowa = linearIndex % a.rows
        val columna = (linearIndex / a.rows) % a.columns
        val rowt = linearIndex % this.rows
        val columnt = linearIndex / this.rows
        prodValues.storage[rowt, columnt] = this.storage[rowt, columnt] * a.storage[rowa, columna]
      }

      this.storage = prodValues.storage

    }



    return this
  }

  /**
   *
   */
  private fun assignProd(a: SparseNDArray): DenseNDArray {

    val newValues: List<Pair<Indices, Double>> = a.map { (indices, value) ->
      Pair(indices, value * this[indices.first, indices.second])
    }

    this.zeros()

    newValues.forEach { (indices, newValue) -> this[indices.first, indices.second] = newValue }

    return this
  }

  /**
   *
   */
  private fun assignProd(a: SparseBinaryNDArray): DenseNDArray {

    val values: List<Pair<Indices, Double>> = a.map { indices -> Pair(indices, this[indices.first, indices.second]) }

    this.zeros()

    values.forEach { (indices, values) -> this[indices.first, indices.second] = values }

    return this
  }

  /**
   *
   */
  override fun div(n: Double): DenseNDArray = DenseNDArray(this.storage.div(n))

  /**
   *
   */
  override fun div(a: NDArray<*>): DenseNDArray = when(a) {
    is DenseNDArray -> DenseNDArray(this.storage.div(a.storage))
    is SparseNDArray -> TODO("not implemented")
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  override fun div(a: SparseNDArray): SparseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }

    return SparseNDArray(
      shape = this.shape.copy(),
      values = DoubleArray(size = a.values.size, init = { i -> this[a.rowIndices[i], a.colIndices[i]] / a.values[i]}),
      rows = a.rowIndices.copyOf(),
      columns = a.colIndices.copyOf()
    )
  }

  /**
   *
   */
  override fun assignDiv(n: Double): DenseNDArray {
    this.storage = this.storage.div(n)
    return this
  }

  /**
   *
   */
  override fun assignDiv(a: DenseNDArray): DenseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }
    this.storage /= a.storage
    return this
  }

  /**
   *
   */
  override fun avg(): Double = this.storage.average()

  /**
   * @return the maximum value of this NDArray
   **/
  override fun max(): Double = this.storage.max()!!

  /**
   * @return the minimum value of this NDArray
   **/
  override fun min(): Double = this.storage.min()!!

  /**
   *
   */
  override fun abs() = DenseNDArray(storage = abs(this.storage))

  /**
   * Sign function.
   *
   * @return a new [DenseNDArray] containing the results of the function sign() applied element-wise
   */
  override fun sign(): DenseNDArray{
    val temp = DenseNDArrayFactory.emptyArray(this.shape)
    for (i in 0 until this.rows)
      for (j in 0 until this.columns)
        temp.storage[i, j] = if(this.storage[i,j]>0.0) 1.0 else if(this.storage[i,j]<0.0) -1.0 else 0.0
    return temp
  }

  /**
   * Non-zero sign function.
   *
   * @return a new [DenseNDArray] containing +1 or -1 values depending on the sign element-wise (+1 if the value is 0)
   */
  fun nonZeroSign(): DenseNDArray{
    val temp = DenseNDArrayFactory.emptyArray(this.shape)
    for (i in 0 until this.rows)
      for (j in 0 until this.columns)
        temp.storage[i,j] = if(this.storage[i,j]>=0.0) 1.0 else -1.0
    return temp
  }
  /**
   *
   */
  override fun sqrt(): DenseNDArray {
    val temp = DenseNDArrayFactory.emptyArray(this.shape)
    for (i in 0 until this.rows)
      for (j in 0 until this.columns)
        temp.storage[i,j] = sqrt(this.storage[i,j])
    return temp
  }

  /**
   *
   */
  override fun assignSqrt(): DenseNDArray {
    for (i in 0 until this.rows)
      for (j in 0 until this.columns)
        this.storage[i,j] = sqrt(this.storage[i,j])
    return this
  }

  /**
   * Square root of this [DenseNDArray] masked by [mask]
   *
   * @param mask the mask to apply
   *
   * @return a [SparseNDArray]
   */
  override fun sqrt(mask: NDArrayMask): SparseNDArray {

    val values = DoubleArray(size = mask.size, init = { sqrt(this.storage[mask.dim1[it], mask.dim2[it]]) })

    return SparseNDArray(shape = this.shape, values = values, rows = mask.dim1, columns = mask.dim2)
  }

  /**
   * Power.
   *
   * @param power the exponent
   *
   * @return a new [DenseNDArray] containing the values of this to the power of [power]
   */
  override fun pow(power: Double): DenseNDArray {
    val temp = DenseNDArrayFactory.emptyArray(this.shape)
    for (i in 0 until this.rows)
      for (j in 0 until this.columns)
        temp.storage[i, j] = this.storage[i, j].pow(power)
    return temp
  }
  /**
   * In-place power.
   *
   * @param power the exponent
   *
   * @return this [DenseNDArray] to the power of [power]
   */
  override fun assignPow(power: Double): DenseNDArray {
    for (i in 0 until this.rows)
      for (j in 0 until this.columns)
        this.storage[i, j] = this.storage[i, j].pow(power)
    return this
  }

  /**
   * Natural exponential.
   *
   * @return a new [DenseNDArray] containing the results of the natural exponential function applied to this
   */
  override fun exp(): DenseNDArray = DenseNDArray(this.storage.exp())

  /**
   * In-place natural exponential.
   *
   * @return this [DenseNDArray] with the natural exponential function applied to its values
   */
  override fun assignExp(): DenseNDArray {
    this.storage = this.storage.exp()
    return this
  }

  /**
   * Logarithm with base 10.
   *
   * @return a new [DenseNDArray] containing the element-wise logarithm with base 10 of this array
   */
  override fun log10(): DenseNDArray {

    require((0 until this.length).all { i -> this[i] != 0.0 })

    return DenseNDArray(this.storage.log()/ ln(10.0))
  }

  /**
   * In-place logarithm with base 10.
   *
   * @return this [DenseNDArray] after having applied the logarithm with base 10 to its values
   */
  override fun assignLog10(): DenseNDArray {

    require((0 until this.length).all { i -> this[i] != 0.0 })

    this.storage = this.storage.log() / ln(10.0)

    return this
  }

  /**
   * Natural logarithm.
   *
   * @return a new [DenseNDArray] containing the element-wise natural logarithm of this array
   */
  override fun ln(): DenseNDArray {

    require((0 until this.length).all { i -> this[i] != 0.0 })

    return DenseNDArray(this.storage.log())
  }

  /**
   * In-place logarithm with base 10.
   *
   * @return this [DenseNDArray] after having applied the logarithm with base 10 to its values
   */
  override fun assignLn(): DenseNDArray {

    require((0 until this.length).all { i -> this[i] != 0.0 })

    this.storage = this.storage.log()

    return this
  }

  /**
   * The norm (L1 distance) of this NDArray.
   *
   * @return the norm
   */
  override fun norm(): Double = linalg.norm(this.storage, Norm.N1)

  /**
   * The Euclidean norm of this DenseNDArray.
   *
   * @return the euclidean norm
   */
  override fun norm2(): Double =
    linalg.norm(this.storage, Norm.Fro)//.distance2(DoubleMatrix.zeros(this.shape.dim1, shape.dim2))

  /**
   * Compute the singular-value decomposition of this DenseNDArray (below called A).
   *
   * @return a triple containing the DenseNDArrays U, S, V such that A = U ⋅ diag(S) ⋅ V'
   */
  @OptIn(org.jetbrains.kotlinx.multik.api.ExperimentalMultikApi::class)
  fun fullSVD(): Triple<DenseNDArray, DenseNDArray, DenseNDArray> {

    val usv = linalg.svd(this.storage)

    return Triple(DenseNDArray(usv.first), DenseNDArrayFactory.eye(usv.second.size, usv.second), DenseNDArray(usv.third))
  }

  /**
   * Compute the sparse variant of the singular-value decomposition of this DenseNDArray (below called A).
   * Sparse means that the matrices U and V are not square but only have as many columns (or rows) as necessary.
   *
   * @return a triple containing the DenseNDArrays U, S, V such that A = U ⋅ diag(S) ⋅ V'
   */
  fun sparseSVD(): Triple<DenseNDArray, DenseNDArray, DenseNDArray> {

    //val usv: Array<DoubleMatrix> = Singular.sparseSVD(this.storage)
    return fullSVD()
    //return Triple(DenseNDArray(usv[0]), DenseNDArray(usv[1]), DenseNDArray(usv[2]))
  }

  /**
   * Get the index of the highest value eventually skipping the element at the given [exceptIndex] when it is >= 0.
   *
   * @param exceptIndex the index to exclude
   *
   * @return the index of the maximum value (-1 if empty)
   **/
  override fun argMaxIndex(exceptIndex: Int): Int {

    var maxIndex: Int = -1
    var maxValue: Double? = null

    (0 until this.length).forEach { i ->

      if (i != exceptIndex) {

        val value = this[i]

        if (maxValue == null || value > maxValue!!) {
          maxValue = value
          maxIndex = i
        }
      }
    }

    return maxIndex
  }

  /**
   * Get the index of the highest value skipping all the elements at the indices in given set.
   *
   * @param exceptIndices the set of indices to exclude
   *
   * @return the index of the maximum value (-1 if empty)
   **/
  override fun argMaxIndex(exceptIndices: Set<Int>): Int {

    var maxIndex: Int = -1
    var maxValue: Double? = null

    (0 until this.length).forEach { i ->

      if (i !in exceptIndices) {

        val value = this[i]

        if (maxValue == null || value > maxValue!!) {
          maxValue = value
          maxIndex = i
        }
      }
    }

    return maxIndex
  }

  /**
   * Round values to Int
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return a new DenseNDArray with the values of the current one rounded to Int
   */
  override fun roundInt(threshold: Double): DenseNDArray {

    val out = DenseNDArrayFactory.emptyArray(this.shape)

    (0 until this.rows * this.columns).forEach { linearIndex ->
      // linear indexing: loop rows before, column by column
      val row = linearIndex % this.rows
      val column = linearIndex / this.rows
      val temp = floor(this.storage[row,column])
      out[row, column] = if (this.storage[row, column]<threshold) temp else temp+1
    }

    return out
  }

  /**
   * Round values to Int in-place
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return this DenseNDArray
   */
  override fun assignRoundInt(threshold: Double): DenseNDArray {


    (0 until this.rows * this.columns).forEach { linearIndex ->
      // linear indexing: loop rows before, column by column
      val row = linearIndex % this.rows
      val column = linearIndex / this.rows
      val temp = floor(this.storage[row,column])
      this.storage[row, column] = if (this.storage[row, column]<threshold) temp else temp+1
    }

    return this
  }

  /**
   * Computes the additive inverse of this nd-array.
   */
  operator fun unaryMinus() = this.prod(-1.0)

  /**
   * Returns the value if this nd-array is a scalar (or 1-element vector/matrix/...).
   */
  fun toScalar(): Double? = if (this.length == 1) this[0] else null

  /**
   * Returns the value if this nd-array is a scalar (or 1-element vector/matrix/...), otherwise throws an exception.
   */
  fun expectScalar(): Double = this.toScalar() ?: throw IllegalStateException("${toString()} is not a scalar")

  /**
   *
   */
  override fun randomize(randomGenerator: RandomGenerator): DenseNDArray {
    for (i in 0 until this.length) this[i] = randomGenerator.next() // i: linear index
    return this
  }

  /**
   *
   */
  override fun concatH(a: DenseNDArray): DenseNDArray = DenseNDArray(this.storage.cat(a.storage, 1))

  /**
   *
   */
  override fun concatV(a: DenseNDArray): DenseNDArray = DenseNDArray(this.storage.cat(a.storage))

  /**
   * Split this NDArray into more NDArrays.
   *
   * If the number of arguments is one, split this NDArray into multiple NDArray each with length [splittingLength].
   * If there are multiple arguments, split this NDArray according to the length of each [splittingLength] element.
   *
   * @param splittingLength the length(s) for sub-array division
   *
   * @return a list containing the split values
   */
  override fun splitV(vararg splittingLength: Int): List<DenseNDArray> =
    if (splittingLength.size == 1)
      this.splitVSingleSegment(splittingLength.first())
    else
      this.splitVMultipleSegments(splittingLength)

  /**
   * Split this NDArray into multiple NDArray each with length [splittingLength]
   *
   * @param splittingLength the length for sub-array division
   *
   * @return a list containing the split values
   */
  private fun splitVSingleSegment(splittingLength: Int): List<DenseNDArray> {

    require(this.length % splittingLength == 0) {
      "The length of the array must be a multiple of the splitting length"
    }

    return List(
      size = this.length / splittingLength,
      init = {
        val startIndex = it * splittingLength
        this.getRange(startIndex, startIndex + splittingLength)
      }
    )
  }

  /**
   * Split this NDArray according to the length of each [splittingLength] element.
   *
   * @param splittingLength the lengths for sub-array division
   *
   * @return a list containing the split values
   */
  private fun splitVMultipleSegments(splittingLength: IntArray): List<DenseNDArray> {

    require(splittingLength.sum() == this.length) {
      "The length of the array must be equal to the sum of each splitting length"
    }

    var offset = 0

    return List(
      size = splittingLength.size,
      init = {
        val startIndex = offset
        offset = startIndex + splittingLength[it]
        this.getRange(startIndex, offset)
      }
    )
  }

  /**
   * @param destination where to store the results of the vectorization
   */
  fun vectorize(destination: DenseNDArray) {

    require(destination.isVector) { "The destination must be a vector." }
    require(destination.length == this.rows * this.columns)

    var i = 0

    for (r in 0 until this.rows) {
      for (c in 0 until this.columns) {
        destination[i++] = this[r, c]
      }
    }
  }

  /**
   *
   */
  fun fromVector(origin: DenseNDArray) {

    require(origin.isVector) { "The origin must be a vector." }
    require(origin.length == this.rows * this.columns)

    var i = 0

    for (r in 0 until this.rows) {
      for (c in 0 until this.columns) {
        this[r, c] = origin[i++]
      }
    }
  }

  /**
   * @param a a DenseNDArray
   * @param tolerance a must be in the range [a - tolerance, a + tolerance] to return True
   *
   * @return a Boolean which indicates if a is equal to be within the tolerance
   */
  override fun equals(a: DenseNDArray, tolerance: Double): Boolean {
    require(this.shape == a.shape)

    return (0 until this.length).all { equals(this[it], a[it], tolerance) }
  }

  /**
   *
   */
  override fun toString(): String = this.storage.toString()

  /**
   *
   */
  override fun equals(other: Any?): Boolean = other is DenseNDArray && this.equals(other)

  /**
   *
   */
  override fun hashCode(): Int = this.storage.hashCode()

  /**
   *
   */
  fun maskBy(mask: NDArrayMask): SparseNDArray = SparseNDArray(
    shape = this.shape,
    values = DoubleArray(size = mask.size, init = { i -> this.storage[mask.dim1[i], mask.dim2[i]] }),
    rows = mask.dim1,
    columns = mask.dim2
  )

  /**
   *
   */
  fun toDoubleArray(): DoubleArray = this.storage.toDoubleArray()

  /**
   * @param reverse whether to sort in descending order
   *
   * @return a permutation of the indices of this array, whose order makes this array sorted.
   */
  fun argSorted(reverse: Boolean = false): IntArray {

    require(this.isVector) { "Operation supported only by vectors." }

    val doubleArray = this.storage.data
    val comparator = Comparator(IndexedDoubleValue::compareTo)
    val indexedValues = Array(doubleArray.size) { IndexedDoubleValue(it, this[it]) }

    indexedValues.sortWith(if (reverse) comparator.reversed() else comparator)
    return IntArray(doubleArray.size) { indexedValues[it].index }
  }

  /**
   * A version of [IndexedValue] specialized to [Double].
   */
  private data class IndexedDoubleValue(val index: Int, val value: Double) : Comparable<IndexedDoubleValue> {

    override fun compareTo(other: IndexedDoubleValue): Int = this.value.compareTo(other.value).let { res ->
      return if (res == 0) index.compareTo(other.index) else res
    }
  }
}
