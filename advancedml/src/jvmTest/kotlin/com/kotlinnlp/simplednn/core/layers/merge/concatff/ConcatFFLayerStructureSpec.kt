/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain contexte at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.merge.concatff

import com.kotlinnlp.simplednn.core.optimizer.ParamsErrorsList
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class ConcatFFLayerStructureSpec : Spek({

  describe("a ConcatLayer") {

    context("forward") {

      val layer = ConcatFFLayerUtils.buildLayer()
      layer.forward()

      it("should match the expected outputArray") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.079830f, -0.669590f, -0.777888f)).equals(
            layer.outputArray.values,
            tolerance = 1.0e-06f)
        }
      }
    }

    context("backward") {

      val layer = ConcatFFLayerUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(ConcatFFLayerUtils.getOutputErrors())

      val paramsErrors: ParamsErrorsList = layer.backward(propagateToInput = true)

      it("should match the expected errors of the weights") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(listOf(
            floatArrayOf(0.625985f, -0.625985f, -0.417323f, -0.069554f, 0.0f, -0.34777f, 0.486877f, 0.486877f, -0.556431f),
            floatArrayOf(0.397187f, -0.397187f, -0.264791f, -0.044132f, 0.0f, -0.22066f, 0.308923f, 0.308923f, -0.353055f),
            floatArrayOf(-0.213241f, 0.213241f, 0.14216f, 0.023693f, 0.0f, 0.118467f, -0.165854f, -0.165854f, 0.189547f)
          )).equals(paramsErrors.getErrorsOf(layer.params.output.unit.weights)!!.values, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the bias") {
        assertTrue {
          DenseNDArrayFactory.arrayOf(floatArrayOf(-0.695539f, -0.441319f, 0.236934f))
            .equals(paramsErrors.getErrorsOf(layer.params.output.unit.biases)!!.values, tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the inputArray at index 0") {
        assertTrue {
          layer.inputArrays[0].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.029384f, 0.109724f, -0.259506f, -0.573413f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the inputArray at index 1") {
        assertTrue {
          layer.inputArrays[1].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.104841f, -0.234286f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the inputArray at index 2") {
        assertTrue {
          layer.inputArrays[2].errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.628016f, 0.295197f, 0.751871f)),
            tolerance = 1.0e-06f)
        }
      }
    }
  }
})
