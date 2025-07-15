/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.functionalities.updatemethods

import com.kotlinnlp.simplednn.core.functionalities.decaymethods.DecayMethod
import com.kotlinnlp.simplednn.core.functionalities.regularization.ParamsRegularization

/**
 * Update methods configuration.
 *
 * @property regularization a parameters regularization method
 */
sealed class UpdateMethodConfig(val regularization: ParamsRegularization? = null) {

  /**
   * AdaGrad configuration.
   *
   * @property learningRate the initial learning rate
   * @property epsilon bias parameter
   * @property regularization a parameters regularization method
   */
  class AdaGradConfig(
    val learningRate: Float = 0.01f,
    val epsilon: Float = 1.0E-8f,
    regularization: ParamsRegularization? = null
  ) : UpdateMethodConfig(regularization)

  /**
   * ADAM configuration.
   *
   * @property stepSize the initial step size
   * @property beta1 the beta1 hyper-parameter
   * @property beta2 the beta2 hyper-parameter
   * @property epsilon the epsilon hyper-parameter
   * @property regularization a parameters regularization method
   */
  class ADAMConfig(
    val stepSize: Float = 0.001f,
    val beta1: Float = 0.9f,
    val beta2: Float = 0.999f,
    val epsilon: Float = 1.0E-8f,
    regularization: ParamsRegularization? = null
  ) : UpdateMethodConfig(regularization)

  /**
   * LearningRate configuration.
   *
   * @property learningRate the initial learning rate
   * @property decayMethod the rate decay method
   * @property regularization a parameters regularization method
   */
  class LearningRateConfig(
    val learningRate: Float,
    val decayMethod: DecayMethod? = null,
    regularization: ParamsRegularization? = null
  ) : UpdateMethodConfig(regularization)

  /**
   * Momentum configuration.
   *
   * @property learningRate the initial learning rate
   * @property momentum  the momentum
   * @property decayMethod the rate decay method
   * @property regularization a parameters regularization method
   */
  class MomentumConfig(
    val learningRate: Float = 0.01f,
    val momentum: Float = 0.9f,
    val decayMethod: DecayMethod? = null,
    regularization: ParamsRegularization? = null
  ) : UpdateMethodConfig(regularization)

  /**
   * NesterovMomentumConfig configuration.

   * @property learningRate the initial learning rate
   * @property momentum  the momentum
   * @property decayMethod the rate decay method
   * @property regularization a parameters regularization method
   */
  class NesterovMomentumConfig(
    val learningRate: Float = 0.01f,
    val momentum: Float = 0.9f,
    val decayMethod: DecayMethod? = null,
    regularization: ParamsRegularization? = null
  ) : UpdateMethodConfig(regularization)

  /**
   * Rectified ADAM configuration.
   *
   * @property stepSize the initial step size
   * @property beta1 the beta1 hyper-parameter
   * @property beta2 the beta2 hyper-parameter
   * @property epsilon the epsilon hyper-parameter
   * @property regularization a parameters regularization method
   */
  class RADAMConfig(
    val stepSize: Float = 0.001f,
    val beta1: Float = 0.9f,
    val beta2: Float = 0.999f,
    val epsilon: Float = 1.0E-8f,
    regularization: ParamsRegularization? = null
  ) : UpdateMethodConfig(regularization)

  /**
   * RMSProp configuration.
   *
   * @property learningRate the initial learning rate
   * @property epsilon a ias parameter
   * @property decay the rate decay parameter
   * @property regularization a parameters regularization method
   */
  class RMSPropConfig(
    val learningRate: Float = 0.001f,
    val epsilon: Float = 1e-08f,
    val decay: Float = 0.95f,
    regularization: ParamsRegularization? = null
  ) : UpdateMethodConfig(regularization)
}
