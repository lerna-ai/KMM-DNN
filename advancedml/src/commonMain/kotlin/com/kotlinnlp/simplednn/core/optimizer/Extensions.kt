/* Copyright 2016-present The KotlinNLP Authors. All Rights Reserved.
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, you can obtain one at http://mozilla.org/MPL/2.0/.
 * ------------------------------------------------------------------*/

package com.kotlinnlp.simplednn.core.optimizer

import com.kotlinnlp.simplednn.core.arrays.ParamsArray

/**
 * @param param a parameters
 *
 * @return the first params errors referring to the given [param], or `null` if no errors refer to it.
 */
fun ParamsErrorsList.getErrorsOf(param: ParamsArray): ParamsArray.Errors<*>? = this.find { it.refParams === param}