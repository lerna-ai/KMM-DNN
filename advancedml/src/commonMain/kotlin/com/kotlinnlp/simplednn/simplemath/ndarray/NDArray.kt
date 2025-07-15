/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArray

/**
 *
 */

interface NDArray<SelfType : NDArray<SelfType>> {

  /**
   *
   */
  val factory: NDArrayFactory<SelfType>

  /**
   *
   */
  val isVector: Boolean

  /**
   * Whether the array is a bi-dimensional matrix
   */
  val isMatrix: Boolean get() = !this.isVector

  /**
   *
   */
  val isOneHotEncoder: Boolean

  /**
   *
   */
  val rows: Int

  /**
   *
   */
  val columns: Int

  /**
   *
   */
  val length: Int

  /**
   *
   */
  val lastIndex: Int

  /**
   *
   */
  val shape: Shape

  /**
   * Transpose
   */
  val t: SelfType

  /**
   *
   */
  operator fun get(i: Int): Number

  /**
   *
   */
  operator fun get(i: Int, j: Int): Number


  /**
   *
   */
  operator fun set(i: Int, value: Number)

  /**
   *
   */
  operator fun set(i: Int, j: Int, value: Number)

  /**
   * Get the i-th row
   *
   * @param i the index of the row to be returned
   *
   * @return the selected row as a new NDArray
   */
  fun getRow(i: Int): SelfType

  /**
   * @return all the rows as a new NDArrays
   */
  fun getRows(): List<SelfType> = (0 until this.rows).map { this.getRow(it) }

  /**
   * Get the i-th column
   *
   * @param i the index of the column to be returned
   *
   * @return the selected column as a new NDArray
   */
  fun getColumn(i: Int): SelfType

  /**
   * @return all the columns as a new NDArrays
   */
  fun getColumns(): List<SelfType> = (0 until this.columns).map { this.getColumn(it) }

  /**
   * Get a one-dimensional NDArray sub-vector of a vertical vector.
   *
   * @param a the start index of the range (inclusive)
   * @param b the end index of the range (exclusive)
   *
   * @return the sub-array
   */
  fun getRange(a: Int, b: Int): SelfType

  /**
   *
   */
  fun zeros(): SelfType

  /**
   *
   */
  fun zerosLike(): SelfType

  /**
   *
   */
  fun copy(): SelfType

  /**
   *
   */
  fun assignValues(n: Float): SelfType

  /**
   *
   */
  fun assignValues(a: NDArray<*>): SelfType

  /**
   *
   */
  fun assignValues(a: NDArray<*>, mask: NDArrayMask): SelfType

  /**
   *
   */
  fun sum(): Float

  /**
   *
   */
  fun sum(n: Float): SelfType

  /**
   *
   */
  fun sum(a: SelfType): SelfType

  /**
   *
   */
  fun sumByRows(a: NDArray<*>): DenseNDArray

  /**
   *
   */
  fun sumByColumns(a: NDArray<*>): DenseNDArray

  /**
   *
   */
  fun assignSum(n: Float): SelfType

  /**
   *
   */
  fun assignSum(a: NDArray<*>): SelfType

  /**
   *
   */
  fun assignSum(a: SelfType, n: Float): SelfType

  /**
   *
   */
  fun assignSum(a: SelfType, b: SelfType): SelfType

  /**
   *
   */
  fun sub(n: Float): SelfType

  /**
   *
   */
  fun sub(a: NDArray<*>): SelfType

  /**
   * In-place subtraction by number
   */
  fun assignSub(n: Float): SelfType

  /**
   *
   */
  fun assignSub(a: NDArray<*>): SelfType

  /**
   *
   */
  fun reverseSub(n: Float): SelfType

  /**
   *
   */
  fun dot(a: NDArray<*>): DenseNDArray

  /**
   *
   */
  fun assignDot(a: SelfType, b: SelfType): SelfType

  /**
   *
   */
  fun assignDot(a: DenseNDArray, b: NDArray<*>): SelfType {
    TODO("not implemented") //To change body of created functions use File | Settings | File Templates.
  }

  /**
   *
   */
  fun prod(n: Float): SelfType

  /**
   *
   */
  fun prod(a: NDArray<*>): SelfType

  /**
   *
   */
  fun prod(n: Float, mask: NDArrayMask): SparseNDArray

  /**
   *
   */
  fun assignProd(n: Float): SelfType

  /**
   *
   */
  fun assignProd(n: Float, mask: NDArrayMask): SelfType

  /**
   *
   */
  fun assignProd(a: NDArray<*>): SelfType

  /**
   *
   */
  fun assignProd(a: SelfType, n: Float): SelfType

  /**
   *
   */
  fun assignProd(a: SelfType, b: SelfType): SelfType

  /**
   *
   */
  fun div(n: Float): SelfType

  /**
   *
   */
  fun div(a: NDArray<*>): SelfType

  /**
   *
   */
  fun div(a: SparseNDArray): SparseNDArray

  /**
   *
   */
  fun assignDiv(n: Float): SelfType

  /**
   *
   */
  fun assignDiv(a: SelfType): SelfType

  /**
   *
   */
  fun abs(): SelfType

  /**
   *
   */
  fun avg(): Float

  /**
   *
   */
  fun max(): Float

  /**
   *
   */
  fun min(): Float

  /**
   * Sign function
   *
   * @return a new NDArray containing the results of the function sign() applied element-wise
   */
  fun sign(): SelfType

  /**
   *
   */
  fun sqrt(): SelfType

  /**
   *
   */
  fun assignSqrt(): SelfType

  /**
   * Square root of this [NDArray] masked by [mask]
   *
   * @param mask the mask to apply
   *
   * @return a [SparseNDArray]
   */
  fun sqrt(mask: NDArrayMask): SparseNDArray

  /**
   * Power.
   *
   * @param power the exponent
   *
   * @return a new [NDArray] containing the values of this to the power of [power]
   */
  fun pow(power: Float): SelfType

  /**
   * In-place power.
   *
   * @param power the exponent
   *
   * @return this [NDArray] to the power of [power]
   */
  fun assignPow(power: Float): SelfType

  /**
   * Natural exponential.
   *
   * @return a new [NDArray] containing the results of the natural exponential function applied to this
   */
  fun exp(): SelfType

  /**
   * In-place natural exponential.
   *
   * @return this [NDArray] with the natural exponential function applied to its values
   */
  fun assignExp(): SelfType

  /**
   * Logarithm with base 10.
   *
   * @return a new [NDArray] containing the element-wise logarithm with base 10 of this array
   */
  fun log10(): SelfType

  /**
   * In-place logarithm with base 10.
   *
   * @return this [NDArray] after having applied the logarithm with base 10 to its values
   */
  fun assignLog10(): SelfType

  /**
   * Natural logarithm.
   *
   * @return a new [NDArray] containing the element-wise natural logarithm of this array
   */
  fun ln(): SelfType

  /**
   * In-place natural logarithm.
   *
   * @return this [NDArray] after having applied the natural logarithm to its values
   */
  fun assignLn(): SelfType

  /**
   * The norm (L1 distance) of this NDArray.
   *
   * @return the norm
   */
  fun norm(): Float

  /**
   * Normalize an array with the L1 distance.
   *
   * @return a new [NDArray] normalized with the L1 distance
   */
  fun normalize(): SelfType {

    val norm: Float = this.norm()

    @Suppress("UNCHECKED_CAST")
    return if (norm != 0.0f) this.div(norm) else this as SelfType
  }

  /**
   * The Euclidean norm of this NDArray.
   *
   * @return the euclidean norm
   */
  fun norm2(): Float

  /**
   * Normalize an array with the Euclidean norm.
   *
   * @return a new [NDArray] normalized with the Euclidean norm
   */
  fun normalize2(): SelfType {

    val norm2: Float = this.norm2()

    @Suppress("UNCHECKED_CAST")
    return if (norm2 != 0.0f) this.div(norm2) else this as SelfType
  }

  /**
   * Get the index of the highest value eventually skipping the element at the given [exceptIndex] when it is >= 0.
   *
   * @param exceptIndex the index to exclude
   *
   * @return the index of the maximum value (-1 if empty)
   **/
  fun argMaxIndex(exceptIndex: Int = -1): Int

  /**
   * Get the index of the highest value skipping all the elements at the indices in given set.
   *
   * @param exceptIndices the set of indices to exclude
   *
   * @return the index of the maximum value (-1 if empty)
   **/
  fun argMaxIndex(exceptIndices: Set<Int>): Int

  /**
   * Round values to Int
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return a new NDArray with the values of the current one rounded to Int
   */
  fun roundInt(threshold: Float = 0.5f): SelfType

  /**
   * Round values to Int in-place
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return this NDArray
   */
  fun assignRoundInt(threshold: Float = 0.5f): SelfType

  /**
   *
   */
  fun randomize(randomGenerator: RandomGenerator): SelfType

  /**
   *
   */
  fun concatH(a: SelfType): SelfType

  /**
   *
   */
  fun concatV(a: SelfType): SelfType

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
  fun splitV(vararg splittingLength: Int): List<SelfType>

  /**
   *
   */
  fun equals(a: SelfType, tolerance: Float = 1.0e-04f): Boolean

  /**
   *
   */
  fun equals(a: Any, tolerance: Float = 1.0e-04f): Boolean {
    @Suppress("UNCHECKED_CAST")
    return a::class.isInstance(this) && this.equals(a as SelfType, tolerance)
  }

  /**
   *
   */
  override fun toString(): String

  /**
   *
   */
  override fun equals(other: Any?): Boolean

  /**
   *
   */
  override fun hashCode(): Int
}
