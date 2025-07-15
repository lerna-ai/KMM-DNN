package com.kotlinnlp.simplednn.core.functionalities.gradientclipping

import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsAccumulator
import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory

object GradientClippingUtils {

  /**
   *
   */
  fun buildParams() = FeedforwardLayerParameters(inputSize = 4, outputSize = 5).also {

    it.unit.weights.values.assignValues(buildDenseParams3())

    it.unit.biases.values.assignValues(buildDenseParams1())
  }

  /**
   *
   */
  fun buildErrors(): ParamsErrorsList {

    val accumulator = ParamsErrorsAccumulator()
    val params = buildParams()

    val gw1 = params.unit.weights.buildDenseErrors(buildWeightsErrorsValues1())
    val gb1 = params.unit.biases.buildDenseErrors(buildBiasesErrorsValues1())
    accumulator.accumulate(listOf(gw1, gb1))

    return accumulator.getParamsErrors()
  }

  /**
   *
   */
  fun buildDenseParams1() = DenseNDArrayFactory.arrayOf(floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f))

  /**
   *
   */
  fun buildDenseParams3() = DenseNDArrayFactory.arrayOf(listOf(
      floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
      floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
      floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
      floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f),
      floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f)
  ))

  /**
   *
   */
  fun buildBiasesErrorsValues1() = DenseNDArrayFactory.arrayOf(floatArrayOf(0.9f, 0.7f, 0.4f, 0.8f, 0.1f))

  /**
   *
   */
  fun buildWeightsErrorsValues1() = DenseNDArrayFactory.arrayOf(listOf(
      floatArrayOf(0.5f, 0.6f, -0.8f, -0.6f),
      floatArrayOf(0.7f, -0.4f, 0.1f, -0.8f),
      floatArrayOf(0.7f, -0.7f, 0.3f, 0.5f),
      floatArrayOf(0.8f, -0.9f, 0.0f, -0.1f),
      floatArrayOf(0.4f, 1.0f, -0.7f, 0.8f)
  ))

}
