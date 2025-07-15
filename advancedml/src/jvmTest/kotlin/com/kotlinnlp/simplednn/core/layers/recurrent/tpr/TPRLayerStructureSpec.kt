/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.layers.recurrent.tpr

import com.kotlinnlp.simplednn.core.functionalities.losses.MSECalculator
import com.kotlinnlp.simplednn.core.optimizer.getErrorsOf
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArray
import com.kotlinnlp.simplednn.simplemath.ndarray.dense.DenseNDArrayFactory
import org.spekframework.spek2.Spek
import org.spekframework.spek2.style.specification.describe
import kotlin.test.assertTrue

class TPRLayerStructureSpec: Spek({

  describe("a TPRLayer") {

    context("forward") {

      context("without previous state context") {

        val layer = TPRLayerStructureUtils.buildLayer(TPRLayersWindow.Empty)
        layer.forward()

        it("should match the expected Symbol Attention Vector") {
          assertTrue {
            layer.aS.values.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.569546f, 0.748381f, 0.509998f, 0.345246f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected Role Attention Vector") {
          assertTrue {
            layer.aR.values.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.291109f, 0.391740f, 0.394126f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected Symbol Vector") {
          assertTrue {
            layer.s.values.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.142810f, 0.913446f, 0.425346f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected Role Vector") {
          assertTrue {
            layer.r.values.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.352204f, 0.205093f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected Output vector") {
          assertTrue {
            layer.outputArray.values.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.050298f, 0.029289f, 0.321719f, 0.187342f, 0.149808f, 0.087235f)),
                tolerance = 0.000001f)
          }
        }
      }

      context("with previous state context") {

        val layer = TPRLayerStructureUtils.buildLayer(TPRLayersWindow.Back)
        layer.forward()

        it("should match the expected Symbol Attention Vector") {
          assertTrue {
            layer.aS.values.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.310412f, 0.880352f, 0.356117f, 0.575599f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected Role Attention Vector") {
          assertTrue {
            layer.aR.values.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.075481f, 0.619886f, 0.357379f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected Symbol Vector") {
          assertTrue {
            layer.s.values.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.169241f, 0.635193f, 0.132934f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected Role Vector") {
          assertTrue {
            layer.r.values.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.323372f, 0.182359f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected Output vector") {
          assertTrue {
            layer.outputArray.values.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.054727f, 0.030862f, 0.205404f, 0.115833f, 0.042987f, 0.024241f)),
                tolerance = 0.000001f)
          }
        }
      }

    }

    context("backward") {

      context("without previous and next state") {

        val layer = TPRLayerStructureUtils.buildLayer(TPRLayersWindow.Empty)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
            output = layer.outputArray.values,
            outputGold = TPRLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.519701f, -0.720710f, 0.471719f, -1.452657f, -0.300191f, -0.022764f)),
                tolerance = 0.00001f)
          }
        }

        it("should match the expected errors of the Binding Matrix") {
          assertTrue {
            layer.bindingMatrix.errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(-0.519701f, -0.720710f),
                    floatArrayOf(0.471719f, -1.452657f),
                    floatArrayOf(-0.300191f, -0.022764f)
                )),
                tolerance = 0.00001f)
          }
        }

        it("should match the expected errors of the Symbol Vector") {
          assertTrue {
            layer.s.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.330854f, -0.131789f, -0.110397f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Role Vector") {
          assertTrue {
            layer.r.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.228986f, -1.439532f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbol attention Vector") {
          assertTrue {
            layer.aS.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.054416f, -0.008955f, -0.021861f, -0.004433f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Role attention Vector") {
          assertTrue {
            layer.aR.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.070328f, -0.052435f, -0.018174f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbols embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.s)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(-0.188436752484536f, -0.247605225547889f, -0.168735193947239f, -0.114226262865736f),
                    floatArrayOf(-0.075060214536815f, -0.098628856128193f, -0.06721247150887f, -0.045499870292789f),
                    floatArrayOf(-0.062876419465944f, -0.082619392545444f, -0.05630252428684f, -0.038114318588475f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Roles embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.r)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(0.06666014408481f, 0.089703290625912f, 0.090249505767454f),
                    floatArrayOf(-0.419062189806341f, -0.563924034648095f, -0.567357842307266f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the input -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(0.043532423752231f, 0.04897397672126f, -0.04897397672126f, -0.005441552969029f),
                    floatArrayOf(0.007164510820115f, 0.00806007467263f, -0.00806007467263f, -0.000895563852514f),
                    floatArrayOf(0.017488998108589f, 0.019675122872163f, -0.019675122872163f, -0.002186124763574f),
                    floatArrayOf(0.003546436137378f, 0.00398974065455f, -0.00398974065455f, -0.000443304517172f)
                )),
                tolerance = 0.000001f)
          }
        }



        it("should match the expected errors of the input -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(0.056263184025661f, 0.063296082028869f, -0.063296082028869f, -0.007032898003208f),
                    floatArrayOf(0.041948301479843f, 0.047191839164823f, -0.047191839164823f, -0.00524353768498f),
                    floatArrayOf(0.014539947335829f, 0.016357440752808f, -0.016357440752808f, -0.001817493416979f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the recurrent -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the recurrent -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Roles bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.070328f, -0.052435f, -0.018174f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbols bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.054416f, -0.008955f, -0.021861f, -0.004433f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.083195466589325f, -0.079995855904333f, -0.000672136225078f, 0.023205789428363f)),
                tolerance = 0.000001f)
          }
        }

      }

      context("with previous state only") {

        val layer = TPRLayerStructureUtils.buildLayer(TPRLayersWindow.Back)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
            output = layer.outputArray.values,
            outputGold = TPRLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.515272047635162f, -0.719137223437654f, 0.355404003016226f, -1.5241663614995f, -0.40701259993718f, -0.085758082190055f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Binding Matrix") {
          assertTrue {
            layer.bindingMatrix.errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(-0.515272047635162f, -0.719137223437654f),
                    floatArrayOf(0.355404003016226f, -1.5241663614995f),
                    floatArrayOf(-0.40701259993718f, -0.085758082190055f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbol Vector") {
          assertTrue {
            layer.s.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.2977662253368f, -0.163018516728468f, -0.147255383603993f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Role Vector") {
          assertTrue {
            layer.r.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.084438933549135f, -1.10124880525326f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbol attention Vector") {
          assertTrue {
            layer.aS.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.052544855142686f, -0.008743664227991f, -0.028606967248956f, 0.009274518622222f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Role attention Vector") {
          assertTrue {
            layer.aR.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.020699695580855f, -0.046236444657814f, -0.019601819976063f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbols embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.s)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(-0.092430456237147f, -0.262139320067201f, -0.106039799492167f, -0.171394125228089f),
                    floatArrayOf(-0.050603038874782f, -0.143513802095011f, -0.058053766198113f, -0.093833395775674f),
                    floatArrayOf(-0.045709960135667f, -0.129636684249611f, -0.052440236745614f, -0.084760142388163f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Roles embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.r)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(0.0063735506599f, 0.052342525049174f, 0.03017676528931f),
                    floatArrayOf(-0.083123563437144f, -0.682648877141492f, -0.393564026977236f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the input -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf( 0.042035884114149f, 0.047290369628418f, -0.047290369628418f, -0.005254485514269f),
                    floatArrayOf(0.006994931382393f, 0.007869297805192f, -0.007869297805192f, -0.000874366422799f),
                    floatArrayOf(0.022885573799165f, 0.02574627052406f, -0.02574627052406f, -0.002860696724896f),
                    floatArrayOf(-0.007419614897777f, -0.00834706676f, 0.00834706676f, 0.000927451862222f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the input -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(0.016559756464684f, 0.01862972602277f, -0.01862972602277f, -0.002069969558086f),
                    floatArrayOf(0.036989155726251f, 0.041612800192032f, -0.041612800192032f, -0.004623644465781f),
                    floatArrayOf(0.01568145598085f, 0.017641637978456f, -0.017641637978456f, -0.001960181997606f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the recurrent -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(-0.011086964435107f, 0.023697729669352f, -0.026219882716201f, 0.070042291905201f, 0.006118848381366f, -0.019231416982223f),
                    floatArrayOf(-0.001844913152106f, 0.003943392566824f, -0.004363088449767f	, 0.011655304415912f, 0.00101819969935f, -0.003200181107445f),
                    floatArrayOf(-0.00603607008953f, 0.012901742229279f, -0.014274876657229f, 0.038133087342858f, 0.003331281336141f, -0.010470150013118f),
                    floatArrayOf(0.001956923429289f, -0.004182807898622f, 0.004627984792489f, -0.012362933323422f, -0.001080017693558f, 0.003394473815733f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the recurrent -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(-0.00436763576756f, 0.009335562706966f, -0.010329148094847f, 0.02759269420928f, 0.002410479550391f, -0.007576088582593f),
                    floatArrayOf(-0.009755889822799f, 0.020852636540674f, -0.023071985884249f, 0.061633180728866f, 0.005384233980402f, -0.01692253874476f),
                    floatArrayOf(-0.004135984014949f, 0.008840420809204f, -0.009781308168055f, 0.026129226028091f, 0.002282631936212f, -0.007174266111239f)
                )),
                tolerance = 0.000001f)
          }
     }

        it("should match the expected errors of the Roles bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.020699695580855f, -0.046236444657814f, -0.019601819976063f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbols bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.052544855142686f, -0.008743664227991f, -0.028606967248956f, 0.009274518622222f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.060099369011985f, -0.048029952866947f, -0.028715724278403f, 0.004889227782339f)),
                tolerance = 0.000001f)
          }
        }

      }

      context("with next state only") {
        val layer = TPRLayerStructureUtils.buildLayer(TPRLayersWindow.Front())

        layer.forward()

        val errors = MSECalculator().calculateErrors(
            output = layer.outputArray.values,
            outputGold = TPRLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.249701403353614f,  0.279289631931703f, 0.98171956873416f, -1.74265783974657f, 0.129808699976926f, 0.747235866903782f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Binding Matrix") {
          assertTrue {
            layer.bindingMatrix.errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(-0.249701403353614f, 0.279289631931703f),
                    floatArrayOf(0.98171956873416f, -1.74265783974657f),
                    floatArrayOf(0.129808699976926f, 0.747235866903782f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbol Vector") {
          assertTrue {
            layer.s.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.030665298337961f, -0.011642597310612f, 0.198972584038394f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Role Vector") {
          assertTrue {
            layer.r.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.916301710420099f, -1.23410486171411f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbol attention Vector") {
          assertTrue {
            layer.aS.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.01567489964703f, 0.007227245531985f, 0.024305171892554f, -0.028759718415032f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Role attention Vector") {
          assertTrue {
            layer.aR.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.000875921596484f, 0.006486559192793f, 0.035967875534f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbols embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.s)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(-0.017465304874356f, -0.022949348763758f, -0.015639261271837f, -0.010587088130656f),
                    floatArrayOf(-0.006630997335104f, -0.008713107019294f, -0.005937709107433f, -0.004019566431043f),
                    floatArrayOf(0.113324083906498f, 0.148907444995259f, 0.101475752605243f, 0.068694596073474f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Roles embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.r)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(0.266744432797764f, 0.358952920168596f, 0.361138631737577f),
                    floatArrayOf(-0.359260053328906f, -0.483449434688323f, -0.48639322191791f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the input -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf( -0.012539919717624f, -0.014107409682327f, 0.014107409682327f, 0.001567489964703f),
                    floatArrayOf(-0.005781796425588f, -0.006504520978787f, 0.006504520978787f, 0.000722724553199f),
                    floatArrayOf(-0.019444137514043f,-0.021874654703299f, 0.021874654703299f, 0.002430517189255f),
                    floatArrayOf(0.023007774732025f, 0.025883746573529f, -0.025883746573529f, -0.002875971841503f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the input -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(0.000700737277188f,  0.000788329436836f, -0.000788329436836f, -8.75921596484405E-05f),
                    floatArrayOf(-0.005189247354234f, -0.005837903273514f, 0.005837903273514f, 0.000648655919279f),
                    floatArrayOf(-0.0287743004272f, -0.0323710879806f, 0.0323710879806f, 0.0035967875534f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the recurrent -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the recurrent -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f),
                    floatArrayOf(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Roles bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.000875921596484f, 0.006486559192793f, 0.035967875534f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbols bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.01567489964703f, 0.007227245531985f, 0.024305171892554f, -0.028759718415032f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.022330619734855f, 0.018660116335174f, 0.045280243815864f, 0.007913525659003f)),
                tolerance = 0.000001f)
          }
        }
      }

      context("with previous and next state") {
        val layer = TPRLayerStructureUtils.buildLayer(TPRLayersWindow.Bilateral)

        layer.forward()

        val errors = MSECalculator().calculateErrors(
            output = layer.outputArray.values,
            outputGold = TPRLayerStructureUtils.getOutputGold())

        layer.outputArray.assignErrors(errors)
        val paramsErrors = layer.backward(propagateToInput = true)

        val params = layer.params

        it("should match the expected errors of the outputArray") {
          assertTrue {
            layer.outputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.245272047635161f, 0.280862776562346f, 0.865404003016226f, -1.8141663614995f, 0.022987400062821f, 0.684241917809945f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Binding Matrix") {
          assertTrue {
            layer.bindingMatrix.errors.equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(-0.245272047635161f, 0.280862776562346f),
                    floatArrayOf(0.865404003016226f, -1.8141663614995f),
                    floatArrayOf(0.022987400062821f, 0.684241917809945f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbol Vector") {
          assertTrue {
            layer.s.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.028096160020637f, -0.050982944923792f, 0.13221154193128f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Role Vector") {
          assertTrue {
            layer.r.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.511244800638788f, -1.01385389080985f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbol attention Vector") {
          assertTrue {
            layer.aS.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.003090437287801f, -0.000276646266682f, 0.010094898398993f, -0.01517018773497f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Role attention Vector") {
          assertTrue {
            layer.aR.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.006956421695052f, -0.011947783124287f, 0.011811289135905f)),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbols embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.s)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(-0.008721408501861f, -0.02473453218537f, -0.010005537638523f, -0.016172138944096f),
                    floatArrayOf(-0.015825760138811f, -0.044882976577447f, -0.018155925008366f, -0.029345763566306f),
                    floatArrayOf(0.041040158690606f, 0.11639279662326f, 0.047082859653024f, 0.076100912884723f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Roles embeddings") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.r)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(0.038589362744448f, 0.316913568882586f, 0.182708541022483f),
                    floatArrayOf(-0.076526891840165f, -0.628473980489548f, -0.362330853963468f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the input -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf( -0.00247234983024f, -0.002781393559021f, 0.002781393559021f, 0.00030904372878f),
                    floatArrayOf(0.000221317013345f	, 0.000248981640013f, -0.000248981640013f, -2.7664626668165E-05f),
                    floatArrayOf(-0.008075918719194f, -0.009085408559093f, 0.009085408559093f, 0.001009489839899f),
                    floatArrayOf(0.012136150187976f, 0.013653168961473f, -0.013653168961473f, -0.001517018773497f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the input -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wInR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(0.005565137356041f,  0.006260779525546f, -0.006260779525546f, -0.000695642169505f),
                    floatArrayOf(0.00955822649943f, 0.010753004811858f, -0.010753004811858f, -0.001194778312429f),
                    floatArrayOf(-0.009449031308724f, -0.010630160222314f, 0.010630160222314f, 0.00118112891359f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the recurrent -> Symbols matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(0.000652082267726f,-0.001393787216798f, 0.001542128206612f, -0.004119552904638f, -0.000359881422164f, 0.001131100047335f),
                    floatArrayOf(-5.83723622698282E-05f, 0.000124767466273f, -0.000138046487074f, 0.000368769473487f, 3.22154577550782E-05f, -0.000101252533605f),
                    floatArrayOf(0.002130023562187f, -0.004552799177946f, 0.005037354301097f, -0.013456499565857f, -0.001175550918563f, 0.003694732814031f),
                    floatArrayOf(-0.003200909612079f, 0.006841754668471f, -0.00756992367975f, 0.020221860250715f, 0.001766568361737f, -0.005552288710999f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the recurrent -> Roles matrix") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.wRecR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(listOf(
                    floatArrayOf(-0.001467804977656f, 0.003137346184468f, -0.003471254425831f, 0.009272910119504f, 0.000810075306389f, -0.002546050340389f),
                    floatArrayOf(-0.002520982239225f, 0.005388450189053f, -0.005961943779019f, 0.015926394904675f, 0.001391319344823f, -0.004372888623489f),
                    floatArrayOf(0.002492182007676f, -0.005326891400293f, 0.005893833278816f, -0.015744448418161f, -0.001375424619876f, 0.004322931823741f)
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Roles bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bR)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.006956421695052f, -0.011947783124287f, 0.011811289135905f
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the Symbols bias") {
          assertTrue {
            (paramsErrors.getErrorsOf(params.bS)!!.values as DenseNDArray).equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(0.003090437287801f, -0.000276646266682f, 0.010094898398993f, -0.01517018773497f
                )),
                tolerance = 0.000001f)
          }
        }

        it("should match the expected errors of the inputArray") {
          assertTrue {
            layer.inputArray.errors.equals(
                DenseNDArrayFactory.arrayOf(floatArrayOf(-0.005503104292945f, -0.005218827534345f, 0.015450218964438f, -0.000914123430928f)),
                tolerance = 0.000001f)
          }
        }
      }
    }
  }
})
