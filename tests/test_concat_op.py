#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest
import numpy as np
from op_test import OpTest


class TestConcatOp(OpTest):
    def setUp(self):
        self.op_type = 'concat'
        self.init_test_data()
        self.inputs = {'X': [('x0', self.x0), ('x1', self.x1), ('x2', self.x2)]}
        self.attrs = {'axis': self.axis}
        self.outputs = {'Out': np.zeros((1, 1)).astype('float32')}

    def test_check_output(self):
        self.check_output()

    def init_test_data(self):
        self.x0 = np.random.random((2, 1, 4, 5)).astype('float32')
        self.x1 = np.random.random((2, 2, 4, 5)).astype('float32')
        self.x2 = np.random.random((2, 3, 4, 5)).astype('float32')
        self.axis = 1


class TestConcatOp2(TestConcatOp):
    def init_test_data(self):
        self.x0 = np.random.random((2, 3, 4, 5)).astype('float32')
        self.x1 = np.random.random((2, 3, 4, 5)).astype('float32')
        self.x2 = np.random.random((2, 3, 4, 5)).astype('float32')
        self.axis = 2


if __name__ == '__main__':
    unittest.main()
