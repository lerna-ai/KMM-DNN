/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.optimizer

import com.kotlinnlp.simplednn.core.functionalities.updatemethods.learningrate.LearningRateMethod
import com.kotlinnlp.simplednn.core.layers.models.feedforward.simple.FeedforwardLayerParameters
import com.kotlinnlp.simplednn.core.optimizer.ParamsOptimizer
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class ParamsOptimizerSpec : Spek({

  describe("a ParamsOptimizer") {

    val learningRateMethod = LearningRateMethod(learningRate = 0.1f)

    context("update after accumulate") {

      val optimizer = ParamsOptimizer(learningRateMethod)

      val params: FeedforwardLayerParameters = ParamsOptimizerUtils.buildParams()

      val gw1 = params.unit.weights.buildDenseErrors(ParamsOptimizerUtils.buildWeightsErrorsValues1())
      val gb1 = params.unit.biases.buildDenseErrors(ParamsOptimizerUtils.buildBiasesErrorsValues1())
      val gw2 = params.unit.weights.buildDenseErrors(ParamsOptimizerUtils.buildWeightsErrorsValues2())
      val gb2 = params.unit.biases.buildDenseErrors(ParamsOptimizerUtils.buildBiasesErrorsValues2())

      optimizer.accumulate(listOf(gw1, gb1, gw2, gb2))
      optimizer.update()

      val w: DenseNDArray = params.unit.weights.values
      val b: DenseNDArray = params.unit.biases.values

      it("should match the expected updated weights") {
        assertTrue {
          w.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(0.2f, 0.44f, 0.17f, -0.12f),
              floatArrayOf(0.1f, -0.15f, 0.18f, 0.56f)
            )),
            tolerance = 1.0e-06f
          )
        }
      }

      it("should match the expected updated biases") {
        assertTrue {
          b.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.36f, -0.37f)),
            tolerance = 1.0e-06f
          )
        }
      }
    }
  }
})
