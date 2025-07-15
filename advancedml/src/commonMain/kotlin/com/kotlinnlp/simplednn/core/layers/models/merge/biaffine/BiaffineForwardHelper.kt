/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.models.merge.biaffine

import com.kotlinnlp.simplednn.core.arrays.ParamsArray
import com.kotlinnlp.simplednn.core.layers.helpers.ForwardHelper
import com.kotlinnlp.simplednn.core.layers.LayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.NDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.Shape
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 * The helper which executes the forward on a [BiaffineLayer].
 *
 * @property layer the layer in which the forward is executed
 */
internal class BiaffineForwardHelper<InputNDArrayType : NDArray<InputNDArrayType>>(
  override val layer: BiaffineLayer<InputNDArrayType>
) : ForwardHelper<InputNDArrayType>(layer) {

  /**
   * Forward the input to the output combining it with the parameters.
   *
   *   wx[ i ] = (wi (dot) x1)' (dot) x2
   *   y = f(wx + w1 (dot) x1 + w2 (dot) x2 + b)
   */
  override fun forward() {

    val x1: InputNDArrayType = this.layer.inputArray1.values
    val x2: InputNDArrayType = this.layer.inputArray2.values

    val wArrays: List<ParamsArray> = this.layer.params.w
    val wx: DenseNDArray = DenseNDArrayFactory.emptyArray(Shape(this.layer.params.outputSize))
    val w1: DenseNDArray = this.layer.params.w1.values
    val w2: DenseNDArray = this.layer.params.w2.values
    val b: DenseNDArray = this.layer.params.b.values

    wArrays.forEachIndexed { i, wArray ->
      val wi: DenseNDArray = wArray.values
      val wx1i: DenseNDArray = this.layer.wx1Arrays[i]

      wx1i.assignDot(wi, x1)
      wx[i] = wx1i.t.dot(x2)[0] // the result is an array with Shape (1, 1)
    }

    this.layer.outputArray.assignValues(w1.dot(x1).assignSum(w2.dot(x2)).assignSum(wx).assignSum(b))

    this.layer.outputArray.activate()
  }
}
