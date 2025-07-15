/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.deeplearning.birnn.utils

import com.kotlinnlp.simplednn.core.functionalities.activations.Tanh
import com.kotlinnlp.simplednn.core.layers.LayerType
import com.kotlinnlp.simplednn.core.layers.models.recurrent.simple.SimpleRecurrentLayerParameters
import com.kotlinnlp.simplednn.deeplearning.birnn.BiRNN
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

/**
 *
 */
object BiRNNEncoderUtils {

  /**
   *
   */
  fun buildInputSequence(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(floatArrayOf(0.5f, 0.6f)),
    DenseNDArrayFactory.arrayOf(floatArrayOf(0.7f, -0.4f)),
    DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, -0.7f))
  )

  /**
   *
   */
  fun buildOutputErrorsSequence(): List<DenseNDArray> = listOf(
    DenseNDArrayFactory.arrayOf(floatArrayOf(-0.4f, -0.8f, 0.1f, 0.4f, 0.6f, -0.4f)),
    DenseNDArrayFactory.arrayOf(floatArrayOf(0.6f, 0.6f, 0.7f, 0.7f, -0.6f, 0.3f)),
    DenseNDArrayFactory.arrayOf(floatArrayOf(-0.1f, -0.1f, 0.1f, -0.8f, 0.4f, -0.5f))
  )

  /**
   *
   */
  fun buildBiRNN(): BiRNN {

    val birnn = BiRNN(
      inputSize = 2,
      inputType = LayerType.Input.Dense,
      hiddenSize = 3,
      hiddenActivation = Tanh,
      recurrentConnectionType = LayerType.Connection.SimpleRecurrent
    )

    this.initL2RParameters(params = birnn.leftToRightNetwork.paramsPerLayer[0] as SimpleRecurrentLayerParameters)
    this.initR2LParameters(params = birnn.rightToLeftNetwork.paramsPerLayer[0] as SimpleRecurrentLayerParameters)

    return birnn
  }

  /**
   *
   */
  private fun initL2RParameters(params: SimpleRecurrentLayerParameters) {

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      floatArrayOf(-0.9f, 0.4f),
      floatArrayOf(0.7f, -1.0f),
      floatArrayOf(-0.9f, -0.4f)
    )))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.4f, -0.3f, 0.8f)))

    params.unit.recurrentWeights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      floatArrayOf(0.1f, 0.9f, -0.5f),
      floatArrayOf(-0.6f, 0.7f, 0.7f),
      floatArrayOf(0.3f, 0.9f, 0.0f)
    )))
  }

  /**
   *
   */
  private fun initR2LParameters(params: SimpleRecurrentLayerParameters) {

    params.unit.weights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      floatArrayOf(0.3f, 0.1f),
      floatArrayOf(0.6f, 0.0f),
      floatArrayOf(-0.7f, 0.1f)
    )))

    params.unit.biases.values.assignValues(DenseNDArrayFactory.arrayOf(floatArrayOf(0.2f, -0.9f, -0.2f)))

    params.unit.recurrentWeights.values.assignValues(DenseNDArrayFactory.arrayOf(listOf(
      floatArrayOf(-0.2f, 0.7f, 0.7f),
      floatArrayOf(-0.2f, 0.0f, -1.0f),
      floatArrayOf(0.5f, -0.4f, 0.4f)
    )))
  }
}
