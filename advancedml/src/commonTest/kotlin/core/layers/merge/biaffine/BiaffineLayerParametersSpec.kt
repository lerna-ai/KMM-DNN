/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain contexte at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package core.layers.merge.biaffine

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.merge.biaffine.BiaffineLayerParameters
import com.nhaarman.mockito_kotlin.mock
import com.nhaarman.mockito_kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals

/**
 *
 */
class BiaffineLayerParametersSpec : Spek({

  describe("a BiaffineLayerParametersS") {

    context("initialization") {

      context("dense input") {

        var k = 0
        val initValues = doubleArrayOf(
          0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1,
          1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = BiaffineLayerParameters(
          inputSize1 = 2,
          inputSize2 = 3,
          outputSize = 2,
          weightsInitializer = RandomInitializer(randomGenerator),
          biasesInitializer = ConstantInitializer(0.9))

        val w1 = params.w1.values
        val w2 = params.w2.values
        val b = params.b.values
        val w = params.w

        it("should contain the expected initialized w1") {
          (0 until w1.length).forEach { i -> assertEquals(initValues[i], w1[i]) }
        }

        it("should contain the expected initialized w2") {
          (0 until w2.length).forEach { i -> assertEquals(initValues[4 + i], w2[i]) }
        }

        it("should contain the expected initialized biases") {
          (0 until b.length).forEach { i -> assertEquals(0.9, b[i]) }
        }

        it("should contain the expected initialized first w array") {
          (0 until w[0].values.length).forEach { i -> assertEquals(initValues[10 + i], w[0].values[i]) }
        }

        it("should contain the expected initialized second w array") {
          (0 until w[1].values.length).forEach { i -> assertEquals(initValues[16 + i], w[1].values[i]) }
        }
      }
    }
  }
})
