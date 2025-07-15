/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Indices
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.SparseEntry
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import com.kotlinnlp.simplednn.simplemath.ndarray.sparse.SparseNDArrayFactory

/**
 *
 */
object UpdateMethodsUtils {

  /**
   *
   */
  fun buildParamsArray(): ParamsArray {

    val values: DenseNDArray = DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, 0.4f, 0.5f, 1.0f, 0.8f))
    val array = ParamsArray(DenseNDArrayFactory.zeros(values.shape))

    array.values.assignValues(values)

    return array
  }

  /**
   *
   */
  fun supportArray1() = DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, 0.8f, 0.5f, 0.3f, 0.2f))

  /**
   *
   */
  fun supportArray2() = DenseNDArrayFactory.arrayOf(floatArrayOf(1.0f, 0.4f, 0.7f, 0.0f, 0.2f))

  /**
   *
   */
  fun buildDenseErrors() = DenseNDArrayFactory.arrayOf(floatArrayOf(0.9f, 0.7f, 0.4f, 0.8f, 0.1f))

  /**
   *
   */
  fun buildSparseErrors() = SparseNDArrayFactory.arrayOf(
    activeIndicesValues = arrayOf(
      SparseEntry(Indices(1, 0), 0.7f),
      SparseEntry(Indices(4, 0), 0.3f)
    ),
    shape = Shape(5)
  )
}
