/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.tpr

import com.kotlinnlp.simplednn.core.functionalities.initializers.ConstantInitializer
import com.kotlinnlp.simplednn.core.functionalities.initializers.RandomInitializer
import com.kotlinnlp.simplednn.core.functionalities.randomgenerators.RandomGenerator
import com.kotlinnlp.simplednn.core.layers.models.recurrent.tpr.TPRLayerParameters
import org.mockito.kotlin.mock
import org.mockito.kotlin.whenever
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertEquals

class TPRLayerParametersSpec: Spek({

  describe("a TPRLayerParameters") {

    context("initialization") {

      context("dense input") {

        var k = 0
        val initValues = floatArrayOf(
            0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f,
            0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f,
            1.3f, 1.4f, 1.5f, 1.6f, 1.7f, 1.8f,
            1.9f, 2.0f, 2.1f, 2.2f, 2.3f, 2.4f,
            2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.0f,
            3.1f, 3.2f, 3.3f, 3.4f, 3.5f, 3.6f,
            3.7f, 3.8f, 3.9f, 4.0f, 4.1f, 4.2f,
            4.3f, 4.4f, 4.5f, 4.6f, 4.7f, 4.8f)
        val randomGenerator = mock<RandomGenerator>()
        whenever(randomGenerator.next()).thenAnswer { initValues[k++] }

        val params = TPRLayerParameters(
            inputSize = 3,
            dRoles = 2,
            dSymbols = 2,
            nRoles = 2,
            nSymbols = 3,
            weightsInitializer = RandomInitializer(randomGenerator),
            biasesInitializer = ConstantInitializer(0.9f))

        val wInS = params.wInS.values
        val wInR = params.wInR.values
        val wRecS = params.wRecS.values
        val wRecR = params.wRecR.values
        val bS = params.bS.values
        val bR = params.bR.values
        val s = params.s.values
        val r = params.r.values

        it("should contain the expected initialized weights of the input -> Symbols matrix") {
          (0 until wInS.length).forEach { i -> assertEquals(initValues[i], wInS[i]) }
        }

        it("should contain the expected initialized weights of the input -> Roles matrix") {
          (0 until wInR.length).forEach { i -> assertEquals(initValues[i + 9], wInR[i]) }
        }

        it("should contain the expected initialized weights of the recurrent -> Symbols matrix") {
          (0 until wRecS.length).forEach { i -> assertEquals(initValues[i + 15], wRecS[i]) }
        }

        it("should contain the expected initialized weights of the recurrent -> Roles matrix") {
          (0 until wRecR.length).forEach { i -> assertEquals(initValues[i + 27], wRecR[i]) }
        }

        it("should contain the expected initialized biases of Symbols") {
          (0 until bS.length).forEach { i -> assertEquals(0.9f, bS[i]) }
        }

        it("should contain the expected initialized biases of Roles") {
          (0 until bR.length).forEach { i -> assertEquals(0.9f, bR[i]) }
        }

        it("should contain the expected initialized weights of the Symbols embeddings matrix") {
          (0 until s.length).forEach { i -> assertEquals(initValues[i + 35], s[i]) }
        }

        it("should contain the expected initialized weights of the Role embeddings matrix") {
          (0 until r.length).forEach { i -> assertEquals(initValues[i + 41], r[i]) }
        }
      }
    }
  }
})
