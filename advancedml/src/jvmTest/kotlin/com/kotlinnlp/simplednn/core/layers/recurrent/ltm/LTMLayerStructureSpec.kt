/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.ltm

import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

/**
 *
 */
class LTMLayerStructureSpec : Spek({

  describe("a LTMLayer") {

    context("forward") {

      context("without previous state context") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayersWindow.Empty)
        layer.forward()

        it("should match the expected input gate L1") {
          assertTrue {
            layer.inputGate1.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.305764f, 0.251618f, 0.574443f, 0.517493f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected input gate L2") {
          assertTrue {
            layer.inputGate2.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.702661f, 0.384616f, 0.244161f, 0.470036f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected input gate L3") {
          assertTrue {
            layer.inputGate3.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.694236f, 0.475021f, 0.731059f, 0.790841f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected cell") {
          assertTrue {
            layer.cell.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.531299f, 0.439948f, 0.484336f, 0.44371f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.368847f, 0.208984f, 0.354078f, 0.350904f)),
              tolerance = 1.0e-06f)
          }
        }
      }

      context("with previous state context") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayersWindow.Back)
        layer.forward()

        it("should match the expected input gate L1") {
          assertTrue {
            layer.inputGate1.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.495f, 0.349781f, 0.372852f, 0.455121f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected input gate L2") {
          assertTrue {
            layer.inputGate2.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.702661f, 0.336261f, 0.334033f, 0.645656f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected input gate L3") {
          assertTrue {
            layer.inputGate3.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.706822f, 0.631812f, 0.547358f, 0.603483f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected cell") {
          assertTrue {
            layer.cell.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.639367f, 0.243846f, 0.477747f, 0.209972f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected outputArray") {
          assertTrue {
            layer.outputArray.values.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.451919f, 0.154065f, 0.261499f, 0.126715f)),
              tolerance = 1.0e-06f)
          }
        }
      }
    }

    context("backward") {

      context("without previous and next state") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayersWindow.Empty)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.201153f, -0.541016f, 0.504078f, -1.289096f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.034775f, -0.063322f, 0.092037f, -0.251637f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.178819f, 0.073623f, 0.263618f, -0.084123f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.026672f, 0.005332f, 0.015735f, -0.009873f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.011423f, 0.004385f, 0.027947f, -0.010844f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.022686f, -0.059356f, 0.048001f, -0.094613f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.021337f, -0.024005f, -0.024005f, 0.026672f),
                floatArrayOf(-0.004266f, -0.004799f, -0.004799f, 0.005332f),
                floatArrayOf(-0.012588f, -0.014161f, -0.014161f, 0.015735f),
                floatArrayOf(0.007898f, 0.008886f, 0.008886f, -0.009873f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.009139f, -0.010281f, -0.010281f, 0.011423f),
                floatArrayOf(-0.003508f, -0.003946f, -0.003946f, 0.004385f),
                floatArrayOf(-0.022357f, -0.025152f, -0.025152f, 0.027947f),
                floatArrayOf(0.008675f, 0.009760f, 0.009760f, -0.010844f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.018149f, 0.020417f, 0.020417f, -0.022686f),
                floatArrayOf(0.047485f, 0.053421f, 0.053421f, -0.059356f),
                floatArrayOf(-0.038401f, -0.043201f, -0.043201f, 0.048001f),
                floatArrayOf(0.075690f, 0.085152f, 0.085152f, -0.094613f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.007471f, -0.003365f, -0.004877f, -0.008459f),
                floatArrayOf(-0.013605f, -0.006128f, -0.008881f, -0.015402f),
                floatArrayOf(0.019774f, 0.008907f, 0.012909f, 0.022387f),
                floatArrayOf(-0.054064f, -0.024353f, -0.035294f, -0.061208f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.226967f, 0.009912f, -0.105134f, -0.040795f)),
              tolerance = 1.0e-06f)
          }
        }
      }

      context("with previous state only") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayersWindow.Back)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.118081f, -0.595935f, 0.411499f, -1.513285f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.019244f, -0.069425f, 0.056198f, -0.151492f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.104782f, 0.054728f, 0.184063f, -0.035139f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.018405f, 0.004185f, 0.014377f, -0.005626f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.010837f, 0.004273f, 0.015267f, -0.003659f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(-0.015645f, -0.033804f, 0.048707f, -0.076034f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.018405f, -0.012883f, -0.022086f, 0.00184f),
                floatArrayOf(-0.004185f, -0.00293f, -0.005023f, 0.000419f),
                floatArrayOf(-0.014377f, -0.010064f, -0.017252f, 0.001438f),
                floatArrayOf(0.005626f, 0.003938f, 0.006751f, -0.000563f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.010837f, -0.007586f, -0.013004f, 0.001084f),
                floatArrayOf(-0.004273f, -0.002991f, -0.005127f, 0.000427f),
                floatArrayOf(-0.015267f, -0.010687f, -0.018320f, 0.001527f),
                floatArrayOf(0.003659f, 0.002561f, 0.004391f, -0.000366f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.015645f, 0.010951f, 0.018774f, -0.001564f),
                floatArrayOf(0.033804f, 0.023663f, 0.040565f, -0.003380f),
                floatArrayOf(-0.048707f, -0.034095f, -0.058449f, 0.004871f),
                floatArrayOf(0.076034f, 0.053224f, 0.091241f, -0.007603f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.019972f, -0.009083f, -0.016466f, -0.015758f),
                floatArrayOf(-0.072048f, -0.032766f, -0.059400f, -0.056847f),
                floatArrayOf(0.058321f, 0.026523f, 0.048083f, 0.046016f),
                floatArrayOf(-0.157217f, -0.071498f, -0.129617f, -0.124046f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.165703f, 0.006373f, -0.085227f, -0.025508f)),
              tolerance = 1.0e-06f)
          }
        }
      }

      context("with next state only") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayersWindow.Front())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.498847f, -0.841016f, 0.304078f, -0.989096f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.08624f, 0.12332f, 0.105471f, -0.316492f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.272226f, 0.109696f, 0.165076f, -0.190172f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.040604f, 0.007945f, 0.009853f, -0.02232f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.017391f, 0.006533f, 0.0175f, -0.024515f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.05626f, -0.09227f, 0.028956f, -0.072595f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.032483f, -0.036544f, -0.036544f, 0.040604f),
                floatArrayOf(-0.006356f, -0.00715f, -0.00715f, 0.007945f),
                floatArrayOf(-0.007882f, -0.008868f, -0.008868f, 0.009853f),
                floatArrayOf(0.017856f, 0.020088f, 0.020088f, -0.02232f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.013912f, -0.015652f, -0.015652f, 0.017391f),
                floatArrayOf(-0.005226f, -0.00588f, -0.005880f, 0.006533f),
                floatArrayOf(-0.014f, -0.01575f, -0.01575f, 0.0175f),
                floatArrayOf(0.019612f, 0.022063f, 0.022063f, -0.024515f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.045008f, -0.050634f, -0.050634f, 0.05626f),
                floatArrayOf(0.073816f, 0.083043f, 0.083043f, -0.09227f),
                floatArrayOf(-0.023165f, -0.026061f, -0.026061f, 0.028956f),
                floatArrayOf(0.058076f, 0.065335f, 0.065335f, -0.072595f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.018529f, 0.008346f, 0.012096f, 0.020977f),
                floatArrayOf(0.026495f, 0.011934f, 0.017296f, 0.029996f),
                floatArrayOf(0.02266f, 0.010207f, 0.014793f, 0.025655f),
                floatArrayOf(-0.067998f, -0.030629f, -0.044390f, -0.076983f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.14514f, 0.004807f, -0.08452f, -0.013372f)),
              tolerance = 1.0e-06f)
          }
        }
      }

      context("with previous and next state") {

        val layer = LTMLayerStructureUtils.buildLayer(LTMLayersWindow.Bilateral)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
          output = layer.outputArray.values,
          outputGold = LTMLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.581919f, -0.895935f, 0.211499f, -1.213285f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the cell") {
          assertTrue {
            layer.cell.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.094839f, 0.061573f, 0.078785f, -0.204401f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of C") {
          assertTrue {
            layer.c.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.180768f, 0.099752f, 0.125337f, -0.114137f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L1") {
          assertTrue {
            layer.inputGate1.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.031751f, 0.007629f, 0.00979f, -0.018275f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L2") {
          assertTrue {
            layer.inputGate2.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.018695f, 0.007787f, 0.010396f, -0.011884f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L3") {
          assertTrue {
            layer.inputGate3.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.0771f, -0.050822f, 0.025034f, -0.060961f)),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L1 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate1.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.031751f, -0.022226f, -0.038102f, 0.003175f),
                floatArrayOf(-0.007629f, -0.00534f, -0.009155f, 0.000763f),
                floatArrayOf(-0.00979f, -0.006853f, -0.011748f, 0.000979f),
                floatArrayOf(0.018275f, 0.012792f, 0.02193f, -0.001827f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L2 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate2.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.018695f, -0.013086f, -0.022434f, 0.001869f),
                floatArrayOf(-0.007787f, -0.005451f, -0.009345f, 0.000779f),
                floatArrayOf(-0.010396f, -0.007277f, -0.012475f, 0.00104f),
                floatArrayOf(0.011884f, 0.008319f, 0.014261f, -0.001188f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the input gate L3 weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.inputGate3.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(-0.0771f, -0.05397f, -0.09252f, 0.00771f),
                floatArrayOf(0.050822f, 0.035575f, 0.060986f, -0.005082f),
                floatArrayOf(-0.025034f, -0.017524f, -0.030041f, 0.002503f),
                floatArrayOf(0.060961f, 0.042673f, 0.073153f, -0.006096f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the cell weights") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.cell.weights)!!.values as DenseNDArray).equals(
              DenseNDArrayFactory.arrayOf(listOf(
                floatArrayOf(0.098423f, 0.044761f, 0.081145f, 0.077657f),
                floatArrayOf(0.0639f, 0.02906f, 0.052682f, 0.050418f),
                floatArrayOf(0.081762f, 0.037183f, 0.067409f, 0.064512f),
                floatArrayOf(-0.212126f, -0.09647f, -0.174887f, -0.16737f)
              )),
              tolerance = 1.0e-06f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
              DenseNDArrayFactory.arrayOf(floatArrayOf(0.065689f, 0.030536f, -0.080868f, -0.011085f)),
              tolerance = 1.0e-06f)
          }
        }
      }
    }
  }
})
