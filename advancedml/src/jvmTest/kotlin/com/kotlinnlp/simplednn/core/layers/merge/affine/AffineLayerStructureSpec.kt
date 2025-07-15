/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain contexte at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.merge.affine

import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class AffineLayerStructureSpec : Spek({

  describe("an AffineLayer") {

    context("forward") {

      val layer = AffineLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.664037f, -0.019997f)),
            tolerance = 1.0e-06f)
        }
      }
    }

    context("backward") {

      val layer = AffineLayerUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(layer.outputArray.values.sub(AffineLayerUtils.getOutputGold()))
      val paramsErrors = layer.backward(propagateToInput = true)

      val params = layer.params

      it("should match the expected errors of the outputArray") {
        assertTrue {
          layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.427139f, 0.279891f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the biases") {
        assertTrue {
          paramsErrors.getErrorsOf(params.b)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.427139f, 0.279891f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of w1") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.w[0])!!.values).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(0.341711f, 0.384425f),
              floatArrayOf(-0.223913f, -0.251902f)
            )),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of w2") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.w[1])!!.values).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(-0.213569f, 0.085428f, -0.256283f),
              floatArrayOf(0.139945f, -0.055978f, 0.167934f)
            )),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the inputArray1") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.095771f, -0.537634f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the inputArray2") {
        assertTrue {
          layer.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.172316f, -0.297537f, 0.468392f)),
            tolerance = 1.0e-06f)
        }
      }
    }
  }
})
