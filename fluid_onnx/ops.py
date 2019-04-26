# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import json
import sys
import onnx
import numpy as np
from functools import partial
from onnx import TensorProto
from onnx.helper import make_node, make_tensor
try:
    from paddle.fluid.executor import fetch_var
except:
    from paddle.fluid.executor import _fetch_var as fetch_var
from fluid.utils import op_io_info, get_old_name
from fluid_onnx.variables import PADDLE_TO_ONNX_DTYPE, paddle_onnx_shape
from compiler.ast import flatten
"""
"""

__onnx_ver__ = onnx.version.version

def fluid_to_onnx_op(operator, block):
    inputs, attrs, outputs = op_io_info(operator)

    if 'op_type' in attrs:
        assert '_op_type' not in attrs
        attrs['_op_type'] = attrs.pop('op_type')

    return make_node(
        operator.type,
        inputs=set(flatten(inputs.values())),
        outputs=set(flatten(outputs.values())),
        inputs_desc=json.dumps(inputs), # save inputs.parameters => inputs.arguments
        outputs_desc=json.dumps(outputs), # save outputs.parameters => outputs.arguments
        **attrs)
