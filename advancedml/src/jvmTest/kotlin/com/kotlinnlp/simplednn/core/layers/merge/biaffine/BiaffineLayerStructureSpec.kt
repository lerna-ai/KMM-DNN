/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain contexte at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.merge.biaffine

import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class BiaffineLayerStructureSpec : Spek({

  describe("a BiaffineLayer") {

    context("forward") {

      val layer = BiaffineLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.714345f, -0.161572f)),
            tolerance = 1.0e-06f)
        }
      }
    }

    context("backward") {

      val layer = BiaffineLayerUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(layer.outputArray.values.sub(BiaffineLayerUtils.getOutputGold()))
      val paramsErrors = layer.backward(propagateToInput = true)

      val params = layer.params

      it("should match the expected errors of the outputArray") {
        assertTrue {
          layer.outputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.398794f, 0.134815f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the biases") {
        assertTrue {
          paramsErrors.getErrorsOf(params.b)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.398794f, 0.134815f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of w1") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.w1)!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(0.319035f, 0.358915f),
              floatArrayOf(-0.107852f, -0.121333f)
            )),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of w2") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.w2)!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(-0.199397f, 0.079759f, -0.239276f),
              floatArrayOf(0.067407f, -0.026963f, 0.080889f)
            )),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the first w array") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.w[0])!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(0.159518f, 0.179457f),
              floatArrayOf(-0.063807f, -0.071783f),
              floatArrayOf(0.191421f, 0.215349f)
            )),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the second w array") {
        assertTrue {
          (paramsErrors.getErrorsOf(params.w[1])!!.values as DenseNDArray).equals(
            DenseNDArrayFactory.arrayOf(listOf(
              floatArrayOf(-0.053926f, -0.060667f),
              floatArrayOf(0.021570f, 0.024267f),
              floatArrayOf(-0.064711f, -0.072800f)
            )),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the inputArray1") {
        assertTrue {
          layer.inputArray1.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(0.048872f, -0.488442f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the inputArray2") {
        assertTrue {
          layer.inputArray2.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.342293f, -0.086394f, 0.601735f)),
            tolerance = 1.0e-06f)
        }
      }
    }
  }
})
