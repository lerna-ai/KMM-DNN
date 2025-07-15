/* Copyright 2020-present Simone Cangialosi. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.feedforward.norm

import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class NormLayerStructureSpec : Spek({

  describe("a NormLayer") {

    context("forward") {

      val layer = NormLayerStructureUtils.buildLayer()
      layer.forward()

      it("should match the expected output") {
        assertTrue {
          layer.outputArray.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(1.15786f, 0.2f, -0.561559f, -0.44465f)),
            tolerance = 1.0e-06f)
        }
      }
    }

    context("backward") {

      val layer = NormLayerStructureUtils.buildLayer()

      layer.forward()

      layer.outputArray.assignErrors(NormLayerStructureUtils.getOutputErrors())

      val paramsErrors = layer.backward(propagateToInput = true)
      val params = layer.params

      it("should match the expected errors of the input") {
        assertTrue {
          layer.inputArray.errors.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.496258f, 0.280667f, -0.408761f, 0.624352f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the weights g") {
        assertTrue {
          paramsErrors.getErrorsOf(params.g)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-0.64465f, -0.25786f, -0.451255f, -0.483487f)),
            tolerance = 1.0e-06f)
        }
      }

      it("should match the expected errors of the bias b") {
        assertTrue {
          paramsErrors.getErrorsOf(params.b)!!.values.equals(
            DenseNDArrayFactory.arrayOf(floatArrayOf(-1.0f, -0.2f, 0.4f, 0.6f)),
            tolerance = 1.0e-06f)
        }
      }
    }
  }
})
