/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward.batchnorm

import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class BatchNormLayerStructureSpec : Spek({

  describe("a BatchNormLayer") {

    context("forward") {

      val layer = BatchNormLayerStructureUtils.buildLayer()
      layer.forward()

      it("should match the expected output at position 0") {
        assertTrue {
          layer.outputArrays[0].values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(1.182833f, 0.2f, -0.519764f, -0.130704f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected output at position 1") {
        assertTrue {
          layer.outputArrays[1].values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.334334f, 0.2f, -0.92716f, -0.571642f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected output at position 2") {
        assertTrue {
          layer.outputArrays[2].values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(1.182833f, 0.2f, -1.253076f, 1.302346f)),
            tolerance = 1.0e-06f)
        }
      }
    }

    context("backward") {

      val layer = BatchNormLayerStructureUtils.buildLayer()

      layer.forward()

      layer.outputArrays[0].assignErrors(BatchNormLayerStructureUtils.getOutputErrors1())
      layer.outputArrays[1].assignErrors(BatchNormLayerStructureUtils.getOutputErrors2())
      layer.outputArrays[2].assignErrors(BatchNormLayerStructureUtils.getOutputErrors3())

      val paramsErrors = layer.backward(propagateToInput = true)
      val params = layer.params

      it("should match the expected errors of the input at position 0") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-1.060623f, 0.0f, -0.325917f, 0.661408f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the input at position 1") {
        assertTrue {
          layer.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.318187f, 0.0f, -0.570354f, 0.992111f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the input at position 2") {
        assertTrue {
          layer.inputArrays[2].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.318187f, 0.0f, -0.570354f, -0.881877f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the weights g") {
        assertTrue {
          paramsErrors.getErrorsOf(params.g)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.070708f, -0.475549f, 0.380236f, -2.218471f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the bias b") {
        assertTrue {
          paramsErrors.getErrorsOf(params.b)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-1.0f, -0.5f, 1.8f, 0.7f)),
            tolerance = 1.0e-06f)
        }
      }
    }
  }
})
