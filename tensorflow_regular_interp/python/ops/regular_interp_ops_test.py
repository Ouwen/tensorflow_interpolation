# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for regular_interp ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.platform import test
from regular_interp_ops import regular_interp


class RegularInterpTest(test.TestCase):

  def testRegularInterp(self):
    with self.test_session():
      self.assertAllClose(regular_interp([[1.0, 2.0]], [1.0, 2.0], [1.5])[0].eval(), 1.5)
      self.assertAllClose(regular_interp([[1.0, 2.0]], [1.0, 2.0], [1.5])[1].eval(), np.array([1.0]))


if __name__ == '__main__':
  test.main()
