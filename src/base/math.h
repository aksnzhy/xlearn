//------------------------------------------------------------------------------
// Copyright (c) 2018 by contributors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//------------------------------------------------------------------------------

/*
This file provides a set of fast ( but approximate )
mathmatic methods.
*/

#ifndef XLEARN_BASE_MATH_H_
#define XLEARN_BASE_MATH_H_

#include <stdlib.h>
#include <stdint.h>

#include <cmath>
#include <random>

#include "src/base/common.h"

typedef float real_t;
typedef uint32 index_t;

/*=====================================================================*
 *                   Copyright (C) 2011 Paul Mineiro                   *
 * All rights reserved.                                                *
 *                                                                     *
 * Redistribution and use in source and binary forms, with             *
 * or without modification, are permitted provided that the            *
 * following conditions are met:                                       *
 *                                                                     *
 *     * Redistributions of source code must retain the                *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer.                                       *
 *                                                                     *
 *     * Redistributions in binary form must reproduce the             *
 *     above copyright notice, this list of conditions and             *
 *     the following disclaimer in the documentation and/or            *
 *     other materials provided with the distribution.                 *
 *                                                                     *
 *     * Neither the name of Paul Mineiro nor the names                *
 *     of other contributors may be used to endorse or promote         *
 *     products derived from this software without specific            *
 *     prior written permission.                                       *
 *                                                                     *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND              *
 * CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,         *
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES               *
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE             *
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER               *
 * OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,                 *
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES            *
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE           *
 * GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR                *
 * BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF          *
 * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT           *
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY              *
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE             *
 * POSSIBILITY OF SUCH DAMAGE.                                         *
 *                                                                     *
 * Contact: Paul Mineiro <paul@mineiro.com>                            *
 *=====================================================================*/

//------------------------------------------------------------------------------
// Fast log()
//------------------------------------------------------------------------------

static inline real_t fastlog2(real_t x) {
  union { real_t f; uint32 i; } vx = { x };
  union { uint32 i; real_t f; } mx = { (vx.i & 0x007FFFFF) | 0x3f000000 };
  real_t y = vx.i;
  y *= 1.1920928955078125e-7f;

  return y - 124.22551499f
           - 1.498030302f * mx.f
           - 1.72587999f / (0.3520887068f + mx.f);
}

static inline real_t fastlog(real_t x) {
  return 0.69314718f * fastlog2 (x);
}

static inline real_t fasterlog2(real_t x) {
  union { real_t f; uint32 i; } vx = { x };
  real_t y = vx.i;
  y *= 1.1920928955078125e-7f;
  return y - 126.94269504f;
}

static inline real_t fasterlog(real_t x){
  union { real_t f; uint32 i; } vx = { x };
  real_t y = vx.i;
  y *= 8.2629582881927490e-8f;
  return y - 87.989971088f;
}

//------------------------------------------------------------------------------
// Fast exp()
//------------------------------------------------------------------------------

static inline real_t fastpow2(real_t p) {
  real_t offset = (p < 0) ? 1.0f : 0.0f;
  real_t clipp = (p < -126) ? -126.0f : p;
  int w = clipp;
  real_t z = clipp - w + offset;
  union { uint32 i; real_t f; } v = {(uint32)
    ((1<<23)*(clipp+121.2740575f+27.7280233f/(4.84252568f-z)-1.49012907f*z))
  };

  return v.f;
}

static inline real_t fastexp(real_t p) {
  return fastpow2(1.442695040f * p);
}

static inline real_t fasterpow2(real_t p) {
  real_t clipp = (p < -126) ? -126.0f : p;
  union { uint32 i; real_t f; } v = {(uint32)
    ((1<<23)*(clipp+126.94269504f))
  };

  return v.f;
}

static inline real_t fasterexp(real_t p) {
  return fasterpow2(1.442695040f * p);
}

//------------------------------------------------------------------------------
// Fast pow()
//------------------------------------------------------------------------------

static inline real_t fastpow(real_t x, real_t p) {
  return fastpow2(p * fastlog2(x));
}

static inline real_t fasterpow(real_t x, real_t p) {
  return fasterpow2(p * fasterlog2(x));
}

//------------------------------------------------------------------------------
// Fast sigmoid()
//------------------------------------------------------------------------------

static inline real_t fastsigmoid(real_t x) {
  return 1.0f / (1.0f + fastexp (-x));
}

static inline real_t fastersigmoid(real_t x) {
  return 1.0f / (1.0f + fasterexp (-x));
}

//------------------------------------------------------------------------------
// 1 / sqrt() Magic function !!
//------------------------------------------------------------------------------
static inline real_t InvSqrt(real_t x) {
  real_t xhalf = 0.5f*x;
  int i = *reinterpret_cast<int*>(&x);  // get bits for floating VALUE
  i = 0x5f375a86-(i>>1);  // gives initial guess y0
  x = *reinterpret_cast<real_t*>(&i);  // convert bits BACK to float
  x = x*(1.5f-xhalf*x*x);  // Newton step, repeating increases accuracy
  return x;
}

#endif   // XLEARN_BASE_MATH_H_
