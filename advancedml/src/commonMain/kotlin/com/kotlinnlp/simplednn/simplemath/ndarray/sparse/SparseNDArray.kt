/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.simplemath.ndarray.sparse

import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.simplemath.equals
import com.kotlinnlp.simplednn.simplemath.ndarray.*
import com.kotlinnlp.simplednn.simplemath.ndarray.sparsebinary.SparseBinaryNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
class SparseNDArray(override val shape: Shape) : NDArray<SparseNDArray>, Iterable<SparseEntry>  {

  companion object {

    /**
     * Private val used to serialize the class (needed by Serializable).
     */
    @Suppress("unused")
    private const val serialVersionUID: Long = 1L

    /**
     * Secondary Factory.
     */
    operator fun invoke(shape: Shape, values: DoubleArray, rows: IntArray, columns: IntArray): SparseNDArray {

      val array = SparseNDArray(shape = shape)

      array.values = values
      array.rowIndices = rows
      array.colIndices = columns

      return array
    }
  }

  /**
   *
   */
  inner class LinearIterator : Iterator<SparseEntry>  {

    /**
     *
     */
    private var curIndex: Int = 0

    /**
     *
     */
    override fun hasNext(): Boolean = this.curIndex < this@SparseNDArray.values.size

    /**
     *
     */
    override fun next(): SparseEntry {

      val value = this@SparseNDArray.values[this.curIndex]
      val indices = Pair(
        this@SparseNDArray.rowIndices[this.curIndex],
        this@SparseNDArray.colIndices[this.curIndex]
      )

      this.curIndex++

      return Pair(indices, value)
    }
  }

  /**
   * Iterator over active indices with the related values
   */
  override fun iterator(): Iterator<SparseEntry> = LinearIterator()

  /**
   *
   */
  var values = doubleArrayOf()
    private set

  /**
   *
   */
  var rowIndices = intArrayOf()
    private set

  /**
   *
   */
  var colIndices = intArrayOf()
    private set

  /**
   *
   */
  override val factory = SparseNDArrayFactory

  /**
   *
   */
  override val isVector: Boolean
    get() = TODO("not implemented")

  /**
   *
   */
  override val isOneHotEncoder: Boolean
    get() = TODO("not implemented")

  /**
   *
   */
  override val rows: Int = this.shape.dim1

  /**
   *
   */
  override val columns: Int = this.shape.dim2

  /**
   *
   */
  override val length: Int = this.rows * this.columns

  /**
   *
   */
  override val lastIndex: Int = this.length - 1

  /**
   * Transpose
   */
  override val t: SparseNDArray get() = SparseNDArray(
    shape = this.shape.inverse,
    values = this.values.copyOf(),
    rows = this.colIndices.copyOf(),
    columns = this.rowIndices.copyOf()
  )

  /**
   *
   */
  val mask: NDArrayMask get() = NDArrayMask(dim1 = this.rowIndices, dim2 = this.colIndices)

  /**
   *
   */
  override fun get(i: Int): Double {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun get(i: Int, j: Int): Double {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun set(i: Int, value: Number) {
    require(i < this.length)

    when {
      this.rows == 1 -> this[0, i] = value
      this.columns == 1 -> this[i, 0] = value
      else -> this[i / this.columns, i % this.columns] = value
    }
  }

  /**
   *
   */
  override fun set(i: Int, j: Int, value: Number) {
    require(i < this.rows && j < this.columns)

    if (value != 0.0) {
      this.setElement(row = i, col = j, value = value.toDouble())

    } else {
      TODO("not implemented")
    }
  }

  /**
   *
   */
  private fun setElement(row: Int, col: Int, value: Double) {

    var index = 0

    while (index < this.values.size && this.colIndices[index] != col) index++
    while (index < this.values.size && this.rowIndices[index] != row) index++

    if (index > this.values.size) {
      throw RuntimeException("Cannot set value at indices not already active")

    } else {
      this.values[index] = value
    }
  }

  /**
   * Get the i-th row
   *
   * @param i the index of the row to be returned
   *
   * @return the selected row as a new [SparseNDArray]
   */
  override fun getRow(i: Int): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Get the i-th column
   *
   * @param i the index of the column to be returned
   *
   * @return the selected column as a new [SparseNDArray]
   */
  override fun getColumn(i: Int): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Get a one-dimensional NDArray sub-vector of a vertical vector.
   *
   * @param a the start index of the range (inclusive)
   * @param b the end index of the range (exclusive)
   *
   * @return the sub-array
   */
  override fun getRange(a: Int, b: Int): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun zeros(): SparseNDArray {
    this.values = doubleArrayOf()
    this.rowIndices = intArrayOf()
    this.colIndices = intArrayOf()
    return this
  }

  /**
   *
   */
  override fun zerosLike(): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun copy(): SparseNDArray = SparseNDArray(
    shape = this.shape.copy(),
    values = this.values.copyOf(),
    rows = this.rowIndices.copyOf(),
    columns = this.colIndices.copyOf()
  )

  /**
   *
   */
  override fun assignValues(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignValues(a: NDArray<*>): SparseNDArray {
    require(a.shape == this.shape)

    return when(a) {
      is DenseNDArray -> TODO("not implemented")
      is SparseNDArray -> this.assignValues(a)
      is SparseBinaryNDArray -> TODO("not implemented")
      else -> throw RuntimeException("Invalid NDArray type")
    }
  }

  /**
   *
   */
  private fun assignValues(a: SparseNDArray): SparseNDArray {

    this.values = a.values.copyOf()
    this.rowIndices = a.rowIndices.copyOf()
    this.colIndices = a.colIndices.copyOf()

    return this
  }

  /**
   *
   */
  fun assignValues(values: DoubleArray, rowIndices: IntArray, colIndices: IntArray): SparseNDArray {
    require(rowIndices.all{ i -> rowIndices[i] in 0 until this.rows}) { "Row index exceeded dim 1" }
    require(colIndices.all{ i -> colIndices[i] in 0 until this.columns}) { "Column index exceeded dim 2" }

    this.values = values.copyOf()
    this.rowIndices = rowIndices.copyOf()
    this.colIndices = colIndices.copyOf()

    return this
  }

  /**
   *
   */
  override fun assignValues(a: NDArray<*>, mask: NDArrayMask): SparseNDArray = when(a) {
    is DenseNDArray -> this.assignValues(a, mask = mask)
    is SparseNDArray -> TODO("not implemented")
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  @Suppress("UNCHECKED_CAST")
  private fun assignValues(a: DenseNDArray, mask: NDArrayMask): SparseNDArray {
    require(this.shape == a.shape)

    this.values = DoubleArray(size = mask.size, init = { k -> a[mask.dim1[k], mask.dim2[k]] })
    this.rowIndices = mask.dim1.copyOf()
    this.colIndices = mask.dim2.copyOf()

    return this
  }

  /**
   *
   */
  override fun sum(): Double = this.values.indices.sumByDouble { i -> this.values[i] }

  /**
   *
   */
  override fun sum(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sum(a: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sumByRows(a: NDArray<*>): DenseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sumByColumns(a: NDArray<*>): DenseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignSum(n: Double): SparseNDArray {

    this.values.indices.forEach { i ->
      this.values[i] += n
    }

    return this
  }

  /**
   *
   */
  override fun assignSum(a: NDArray<*>): SparseNDArray = when(a) {
    is DenseNDArray -> TODO("not implemented")
    is SparseNDArray -> this.assignSum(a)
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   * Sum the values of [a] to this [SparseNDArray], updating the arrays [values], [rowIndices] and [colIndices] with all
   * the entries which are in [a] and not in this [SparseNDArray] and adding the values of [a] at indices already active
   * in this.
   *
   * @param a the [SparseNDArray] to add to this
   *
   * @return this [SparseNDArray]
   */
  @Suppress("UNCHECKED_CAST")
  private fun assignSum(a: SparseNDArray): SparseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }

    fun SparseNDArray.getLinearIndex(arrayIndex: Int): Int =
      this.colIndices[arrayIndex] * this.shape.dim1 + this.rowIndices[arrayIndex]

    val thisSize = this.values.size
    val aSize = a.values.size
    val concatSize = thisSize + aSize

    val reducedValues = DoubleArray(size = concatSize)
    val reducedRows = IntArray(size = concatSize)
    val reducedCols = IntArray(size = concatSize)

    var k = 0
    var aK = 0
    var lastIndex = -1

    for (i in 0 until concatSize) {
      val thisRef: Boolean = aK >= aSize || (k < thisSize && this.getLinearIndex(k) < a.getLinearIndex(aK))
      val ref: SparseNDArray = if (thisRef) this else a
      val index: Int = if (thisRef) k++ else aK++

      if (lastIndex >= 0
        && ref.rowIndices[index] == reducedRows[lastIndex]
        && ref.colIndices[index] == reducedCols[lastIndex]) {

        reducedValues[lastIndex] = reducedValues[lastIndex] + ref.values[index]

      } else {
        lastIndex++
        reducedValues[lastIndex] = ref.values[index]
        reducedRows[lastIndex] = ref.rowIndices[index]
        reducedCols[lastIndex] = ref.colIndices[index]
      }
    }

    this.values = reducedValues.copyOfRange(0, lastIndex + 1)
    this.rowIndices = reducedRows.copyOfRange(0, lastIndex + 1)
    this.colIndices= reducedCols.copyOfRange(0, lastIndex + 1)

    return this
  }

  /**
   *
   */
  override fun assignSum(a: SparseNDArray, n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignSum(a: SparseNDArray, b: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sub(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sub(a: NDArray<*>): SparseNDArray = when(a) {
    is DenseNDArray -> TODO("not implemented")
    is SparseNDArray -> TODO("not implemented")
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   * In-place subtraction by number
   */
  override fun assignSub(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignSub(a: NDArray<*>): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun reverseSub(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun dot(a: NDArray<*>): DenseNDArray = when(a) {
    is DenseNDArray -> this.dot(a)
    is SparseNDArray -> TODO("not implemented")
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  private fun dot(a: DenseNDArray): DenseNDArray {
    require(this.columns == a.rows)

    val ret: DenseNDArray = DenseNDArrayFactory.zeros(Shape(this.rows, a.columns))

    for (aCol in 0 until a.shape.dim2) {
      this.values.indices.forEach { k ->
        val row: Int = this.rowIndices[k]
        val col: Int = this.colIndices[k]

        ret[row, aCol] += this.values[k] * a[col, aCol]
      }
    }

    return ret
  }

  /**
   *
   */
  override fun assignDot(a: SparseNDArray, b: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignDot(a: DenseNDArray, b: NDArray<*>): SparseNDArray {

    when(b) {
      is DenseNDArray -> TODO("not implemented")
      is SparseNDArray -> TODO("not implemented")
      is SparseBinaryNDArray -> this.assignDot(a, b)
    }

    return this
  }

  /**
   *
   */
  @Suppress("UNCHECKED_CAST")
  fun assignDot(a: SparseNDArray, b: DenseNDArray): SparseNDArray {
    require(a.rows == this.rows && b.columns == this.columns && a.columns == b.rows)
    require(a.columns == 1 && b.rows == 1) // TODO: extend to all shapes

    val resultValuesSize = a.values.size * b.length
    val activeRowIndicesSize: Int = a.rowIndices.size

    val cols = IntArray(size = resultValuesSize, init = { k -> k / activeRowIndicesSize })

    this.values = DoubleArray(size = resultValuesSize, init = { k -> a.values[k % activeRowIndicesSize] * b[cols[k]] })
    this.rowIndices = IntArray(size = resultValuesSize, init = { k -> a.rowIndices[k % activeRowIndicesSize] })
    this.colIndices = cols

    return this
  }

  /**
   *
   */
  @Suppress("UNCHECKED_CAST")
  private fun assignDot(a: DenseNDArray, b: SparseBinaryNDArray): SparseNDArray {
    require(a.rows == this.rows) { "a.rows (%d) != this.rows (%d)".format(a.rows, this.rows) }
    require(b.columns == this.columns) { "b.columns (%d) != this.columns (%d)".format(b.columns, this.columns) }
    require(a.columns == b.rows) { "a.columns (%d) != b.rows (%d)".format(a.columns, b.rows) }

    when {
      b.rows == 1 -> {
        // Column vector (dot) row vector

        val bActiveIndices = b.activeIndicesByColumn.keys
        val valuesCount = bActiveIndices.size * a.rows
        val values = DoubleArray(size = valuesCount)
        val rows = IntArray(size = valuesCount)
        val columns = IntArray(size = valuesCount)

        var k = 0
        for (j in bActiveIndices) {
          for (i in 0 until a.rows) {
            values[k] = a[i]
            rows[k] = i
            columns[k] = j
            k++
          }
        }

        this.values = values
        this.rowIndices = rows
        this.colIndices = columns

      }
      b.columns == 1 -> {
        // n-dim array (dot) column vector
        this.zeros()
        this.values = DoubleArray(size = a.rows, init = { i -> b.activeIndicesByRow.keys.sumByDouble { a[i, it] } })
        this.rowIndices = IntArray(size = a.rows, init = { it })
        this.colIndices = IntArray(size = a.rows, init = { 0 })


      }
      else -> // n-dim array (dot) n-dim array
        TODO("not implemented")
    }

    return this
  }

  /**
   *
   */
  override fun prod(n: Double) = SparseNDArray(
    shape = this.shape,
    values = DoubleArray(size = this.values.size, init = { i -> this.values[i] * n }),
    rows = this.rowIndices.copyOf(),
    columns = this.colIndices.copyOf()
  )

  /**
   *
   */
  override fun prod(a: NDArray<*>): SparseNDArray = when(a) {
    is DenseNDArray -> this.prod(a)
    is SparseNDArray -> this.prod(a)
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  private fun prod(a: SparseNDArray): SparseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }
    require(a.values.size == this.values.size) { "Arrays with a different amount of active values" }

    return SparseNDArray(
      shape = this.shape,
      values = DoubleArray(size = this.values.size, init = { i -> this.values[i] * a.values[i]}),
      rows = this.rowIndices.copyOf(),
      columns = this.colIndices.copyOf())
  }

  /**
   * Product by a [DenseNDArray] with the same shape or a compatible column vector (each column is multiplied
   * by the given vector).
   *
   * @param a the [DenseNDArray] by which this [SparseNDArray] will be multiplied
   *
   * @return a new [SparseNDArray] containing the product between this [SparseNDArray] and [a]
   */
  private fun prod(a: DenseNDArray): SparseNDArray {
    require(a.shape == this.shape || (a.columns == 1 && a.rows == this.rows)) { "Arrays with not compatible size" }

    return if (a.shape == this.shape)
      SparseNDArray(
        shape = this.shape,
        values = DoubleArray(
          size = this.values.size,
          init = { k -> this.values[k] * a[this.rowIndices[k], this.colIndices[k]] }
        ),
        rows = this.rowIndices.copyOf(),
        columns = this.colIndices.copyOf())

    else
      SparseNDArray(
        shape = this.shape,
        values = DoubleArray(
          size = this.values.size,
          init = { k -> this.values[k] * a[this.rowIndices[k], 0] }
        ),
        rows = this.rowIndices.copyOf(),
        columns = this.colIndices.copyOf())
  }

  /**
   *
   */
  override fun prod(n: Double, mask: NDArrayMask): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(n: Double): SparseNDArray {

    this.values.indices.forEach { i ->
      this.values[i] *= n
    }

    return this
  }

  /**
   *
   */
  override fun assignProd(n: Double, mask: NDArrayMask): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(a: SparseNDArray, n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(a: SparseNDArray, b: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignProd(a: NDArray<*>): SparseNDArray = when(a) {
    is DenseNDArray -> TODO("not implemented")
    is SparseNDArray -> this.assignProd(a)
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  private fun assignProd(a: SparseNDArray): SparseNDArray {

    require(a.shape == this.shape) { "Arrays with different size" }
    require(a.values.size == this.values.size) { "Arrays with a different amount of active values" }

    this.values.indices.forEach { i ->
      this.values[i] *= a.values[i]
    }

    return this
  }

  /**
   *
   */
  override fun div(n: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun div(a: SparseNDArray): SparseNDArray {
    require(a.shape == this.shape) { "Arrays with different size" }
    require(a.values.size == this.values.size) { "Arrays with a different amount of active values" }

    return SparseNDArray(
      shape = this.shape,
      values = DoubleArray(size = this.values.size, init = { i -> this.values[i] / a.values[i]}),
      rows = this.rowIndices.copyOf(),
      columns = this.colIndices.copyOf())
  }

  /**
   *
   */
  override fun div(a: NDArray<*>): SparseNDArray = when(a) {
    is DenseNDArray -> TODO("not implemented")
    is SparseNDArray -> this.div(a)
    is SparseBinaryNDArray -> TODO("not implemented")
    else -> throw RuntimeException("Invalid NDArray type")
  }

  /**
   *
   */
  override fun assignDiv(n: Double): SparseNDArray {

    this.values.indices.forEach { i ->
      this.values[i] /= n
    }

    return this
  }

  /**
   *
   */
  override fun assignDiv(a: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun avg(): Double {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun abs(): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Sign function
   *
   * @return a new [SparseNDArray] containing the results of the function sign() applied element-wise
   */
  override fun sign(): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun sqrt(): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun assignSqrt(): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Square root of this [SparseNDArray] masked by [mask]
   *
   * @param mask the mask to apply
   *
   * @return a [SparseNDArray]
   */
  override fun sqrt(mask: NDArrayMask): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Power.
   *
   * @param power the exponent
   *
   * @return a new [SparseNDArray] containing the values of this to the power of [power]
   */
  override fun pow(power: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * In-place power.
   *
   * @param power the exponent
   *
   * @return this [SparseNDArray] to the power of [power]
   */
  override fun assignPow(power: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Natural exponential.
   *
   * @return a new [SparseNDArray] containing the results of the natural exponential function applied to this
   */
  override fun exp(): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * In-place natural exponential.
   *
   * @return this [SparseNDArray] with the natural exponential function applied to its values
   */
  override fun assignExp(): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Logarithm with base 10.
   *
   * @return a new [SparseNDArray] containing the element-wise logarithm with base 10 of this array
   */
  override fun log10(): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * In-place logarithm with base 10.
   *
   * @return this [SparseNDArray] after having applied the logarithm with base 10 to its values
   */
  override fun assignLog10(): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Natural logarithm.
   *
   * @return a new [SparseNDArray] containing the element-wise natural logarithm of this array
   */
  override fun ln(): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * In-place logarithm with base 10.
   *
   * @return this [SparseNDArray] after having applied the logarithm with base 10 to its values
   */
  override fun assignLn(): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * The norm (L1 distance) of this NDArray.
   *
   * @return the norm
   */
  override fun norm(): Double {
    TODO("not implemented")
  }

  /**
   * The Euclidean norm of this DenseNDArray.
   *
   * @return the euclidean norm
   */
  override fun norm2(): Double {
    TODO("not implemented")
  }

  /**
   * @return the maximum value of this NDArray
   **/
  override fun max(): Double = this.values.max()!!

  /**
   * @return the minimum value of this NDArray
   **/
  override fun min(): Double = this.values.min()!!

  /**
   * Get the index of the highest value eventually skipping the element at the given [exceptIndex] when it is >= 0.
   *
   * @param exceptIndex the index to exclude
   *
   * @return the index of the maximum value (-1 if empty)
   */
  override fun argMaxIndex(exceptIndex: Int): Int {
    TODO("not implemented")
  }

  /**
   * Get the index of the highest value skipping all the elements at the indices in given set.
   *
   * @param exceptIndices the set of indices to exclude
   *
   * @return the index of the maximum value (-1 if empty)
   **/
  override fun argMaxIndex(exceptIndices: Set<Int>): Int {
    TODO("not implemented")
  }

  /**
   * Round values to Int
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return a new NDArray with the values of the current one rounded to Int
   */
  override fun roundInt(threshold: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Round values to Int in-place
   *
   * @param threshold a value is rounded to the next Int if is >= [threshold], to the previous otherwise
   *
   * @return this [SparseNDArray]
   */
  override fun assignRoundInt(threshold: Double): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun randomize(randomGenerator: RandomGenerator): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun concatH(a: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun concatV(a: SparseNDArray): SparseNDArray {
    TODO("not implemented")
  }

  /**
   * Splits this NDArray into more NDArrays.
   *
   * If the number of arguments is one, split this NDArray into multiple NDArray each with length [splittingLength].
   * If there are multiple arguments, split this NDArray according to the length of each [splittingLength] element.
   *
   * @param splittingLength the length(s) for sub-array division
   *
   * @return a list containing the split values
   */
  override fun splitV(vararg splittingLength: Int): List<SparseNDArray> {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun equals(a: SparseNDArray, tolerance: Double): Boolean {

    this.sortValues()
    a.sortValues()

    return equals(this.values, a.values, tolerance = tolerance) &&
      this.rowIndices.contentEquals(a.rowIndices) &&
      this.colIndices.contentEquals(a.colIndices)
  }

  /**
   *
   */
  override fun toString(): String {
    TODO("not implemented")
  }

  /**
   *
   */
  override fun equals(other: Any?): Boolean = other is SparseNDArray && this.equals(other)

  /**
   *
   */
  override fun hashCode(): Int {
    TODO("not implemented")
  }

  /**
   *
   */
  private fun sortValues() {
    if (this.colIndices.size > 1) {
      this.quicksort(0, this.colIndices.lastIndex)
    }
  }

  /**
   *
   */
  private fun quicksort(lo: Int, hi: Int) {

    if (lo < hi) {
      val p: Int = this.partition(lo, hi)
      this.quicksort(lo, p - 1)
      this.quicksort(p + 1, hi)
    }
  }

  /**
   *
   */
  private fun partition(lo: Int, hi: Int): Int {

    val pivot: Int = hi
    var i: Int = lo

    while (i < pivot && this.compareArrays(i, pivot) <= 0) i++

    ((i + 1) until hi).forEach { j ->
      if (this.compareArrays(j, pivot) <= 0) {
        this.swapArrays(i++, j)
      }
    }

    this.swapArrays(i, pivot)

    return i
  }

  /**
   *
   */
  private fun compareArrays(i: Int, j: Int): Int {
    return if (this.colIndices[i] != this.colIndices[j])
      this.colIndices[i] - this.colIndices[j]
    else
      this.rowIndices[i] - this.rowIndices[j]
  }

  /**
   *
   */
  private fun swapArrays(i: Int, j: Int) {
    if (i != j) {
      this.swapArray(this.values, i, j)
      this.swapArray(this.rowIndices, i, j)
      this.swapArray(this.colIndices, i, j)
    }
  }

  /**
   *
   */
  private fun swapArray(array: IntArray, i: Int, j: Int) {
    val tmp: Int = array[i]
    array[i] = array[j]
    array[j] = tmp
  }

  /**
   *
   */
  private fun swapArray(array: DoubleArray, i: Int, j: Int) {
    val tmp: Double = array[i]
    array[i] = array[j]
    array[j] = tmp
  }
}
