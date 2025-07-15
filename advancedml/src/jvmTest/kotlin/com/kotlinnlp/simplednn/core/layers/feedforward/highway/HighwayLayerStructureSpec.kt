/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain contexte at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward.highway

import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class HighwayLayerStructureSpec : Spek({

  describe("a HighwayLayer") {

    context("forward") {

      val layer = HighwayLayerStructureUtils.buildLayer()
      layer.forward()

      it("should match the expected input unit") {
        assertTrue {
          layer.inputUnit.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.39693f, -0.796878f, 0.0f, 0.701374f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected transform gate") {
        assertTrue {
          layer.transformGate.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.85321f, 0.432907f, 0.116089f, 0.519989f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.456097f, -0.855358f, -0.79552f, 0.844718f)),
            tolerance = 1.0e-06f)
        }
      }
    }

    context("backward") {

      val layer = HighwayLayerStructureUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(HighwayLayerStructureUtils.getOutputErrors())

      val paramsErrors = layer.backward(propagateToInput = true)
      val params = layer.params

      it("should match the expected errors of the outputArray") {
        assertTrue {
          layer.outputArray.errors.equals(
            HighwayLayerStructureUtils.getOutputErrors(),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the input unit") {
        assertTrue {
          layer.inputUnit.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.409706f, 0.118504f, -0.017413f, 0.433277f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the transform gate") {
        assertTrue {
          layer.transformGate.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.028775f, 0.018987f, -0.013853f, -0.122241f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the input unit biases") {
        assertTrue {
          paramsErrors.getErrorsOf(params.input.biases)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.409706f, 0.118504f, -0.017413f, 0.433277f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the transform gate biases") {
        assertTrue {
          paramsErrors.getErrorsOf(params.transformGate.biases)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.028775f, 0.018987f, -0.013853f, -0.122241f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the input unit weights") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.input.weights)!!.values).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(-0.327765f, -0.368736f, -0.368736f, 0.409706f),
              floatArrayOf(-0.094803f, -0.106653f, -0.106653f, 0.118504f),
              floatArrayOf(0.013931f, 0.015672f, 0.015672f, -0.017413f),
              floatArrayOf(-0.346622f, -0.389949f, -0.389949f, 0.433277f)
            )),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the transform gate weights") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.transformGate.weights)!!.values).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(-0.023020f, -0.025897f, -0.025897f, 0.028775f),
              floatArrayOf(-0.015190f, -0.017088f, -0.017088f, 0.018987f),
              floatArrayOf(0.011082f, 0.012467f, 0.012467f, -0.013853f),
              floatArrayOf(0.097793f, 0.110017f, 0.110017f, -0.122241f)
            )),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the inputArray") {
        assertTrue {
          layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.822397f, 0.132596f, -0.437003f, 0.446894f)),
            tolerance = 1.0e-06f)
        }
      }
    }
  }
})
