/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.attention.scaleddot

import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.attention.scaleddot.ScaledDotAttentionLayerParameters
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class ScaledDotAttentionParametersSpec : Spek({

  describe("ScaledDotAttentionParameters") {

    context("initialization") {

      val randomGenerator = mock<RandomGenerator>()
      var i = 0.0f
      whenever(randomGenerator.next()).thenAnswer { i++ }

      val params = ScaledDotAttentionLayerParameters(
        inputSize = 2,
        attentionSize = 3,
        outputSize = 2,
        weightsInitializer = RandomInitializer(randomGenerator),
        biasesInitializer = RandomInitializer(randomGenerator))

      it("should have the queries weights with the expected initialized values") {
        assertTrue {
          params.queries.weights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(0.0f, 3.0f),
              floatArrayOf(1.0f, 4.0f),
              floatArrayOf(2.0f, 5.0f)
            )),
            tolerance = 1.0e-06f
          )
        }
      }

      it("should have the keys weights with the expected initialized values") {
        assertTrue {
          params.keys.weights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(6.0f, 9.0f),
              floatArrayOf(7.0f, 10.0f),
              floatArrayOf(8.0f, 11.0f)
            )),
            tolerance = 1.0e-06f
          )
        }
      }

      it("should have the values weights with the expected initialized values") {
        assertTrue {
          params.values.weights.values.equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(12.0f, 14.0f),
              floatArrayOf(13.0f, 15.0f)
            )),
            tolerance = 1.0e-06f
          )
        }
      }

      it("should have the queries biases with the expected initialized values") {
        assertTrue {
          params.queries.biases.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(16.0f, 17.0f, 18.0f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should have the keys biases with the expected initialized values") {
        assertTrue {
          params.keys.biases.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(19.0f, 20.0f, 21.0f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should have the values biases with the expected initialized values") {
        assertTrue {
          params.values.biases.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(22.0f, 23.0f)),
            tolerance = 1.0e-06f)
        }
      }
    }
  }
})
