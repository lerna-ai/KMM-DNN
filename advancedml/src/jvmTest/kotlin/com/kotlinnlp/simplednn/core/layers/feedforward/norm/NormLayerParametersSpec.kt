/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward.norm

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.feedforward.norm.NormLayerParameters
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals

/**
 *
 */
class NormLayerParametersSpec: Spek({

  describe("NormLayerParameters") {

    context("initialization") {

      context("dense input") {

        var k = 0
        val initValues = floatArrayOf(0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = NormLayerParameters(
          inputSize = 10,
          weightsInitializer = RandomInitializer(randomGenerator),
          biasesInitializer = ConstantInitializer(0.9f))

        it("should contain the expected initialized weights g") {
          (0 until params.g.values.length).forEach { i ->
            assertEquals(initValues[i], params.g.values[i])
          }
        }

        it("should contain the expected initialized biases b") {
          (0 until params.b.values.length).forEach { i ->
            assertEquals(0.9f, params.b.values[i])
          }
        }
      }
    }
  }
})
