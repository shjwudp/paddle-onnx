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
from onnx.helper import get_attribute_value, make_attribute


def rename_node_input(
        onnx_node, # type: onnx.NodeProto
        src_value_name, # type: str
        dst_value_name # type: str
): # type: (...) -> None
    """
    rename onnx_node input
    and attribute["inputs_desc"] for fluid converted onnx node
    """
    # rename onnx_node.input
    onnx_node.input[:] = [
        dst_value_name if x == src_value_name else x for x in onnx_node.input]

    # rename onnx_node.attribute["inputs_desc"]
    A = filter(lambda x: x.name == 'inputs_desc', onnx_node.attribute)
    assert len(A) <= 1, 'len(A)={}'.format(len(A))
    if len(A) == 1:
        inputs_desc_pb = A[0]
        inputs_desc = json.loads(get_attribute_value(inputs_desc_pb))
        for input_name in inputs_desc:
            inputs_desc[input_name] = [
                dst_value_name if x == src_value_name else x for x in inputs_desc[input_name]]
        inputs_desc_pb.CopyFrom(make_attribute('inputs_desc', json.dumps(inputs_desc)))


def rename_node_output(
        onnx_node, # type: onnx.NodeProto
        src_value_name, # type: str
        dst_value_name # type: str
): # type: (...) -> None
    """
    rename onnx_node output
    and attribute["outputs_desc"] for fluid converted onnx node
    """
    # rename onnx_node.output
    onnx_node.output[:] = [
        dst_value_name if x == src_value_name else x for x in onnx_node.output]

    # rename onnx_node.attribute["outputs_desc"]
    A = filter(lambda x: x.name == 'outputs_desc', onnx_node.attribute)
    assert len(A) <= 1, 'len(A)={}'.format(len(A))
    if len(A) == 1:
        outputs_desc_pb = A[0]
        outputs_desc = json.loads(get_attribute_value(outputs_desc_pb))
        for output_name in outputs_desc:
            outputs_desc[output_name] = [
                dst_value_name if x == src_value_name else x for x in outputs_desc[output_name]]
        outputs_desc_pb.CopyFrom(make_attribute('outputs_desc', json.dumps(outputs_desc)))


def rename_value(
        onnx_graph, # type: onnx.GraphProto
        src_value_name, # type: str
        dst_value_name # type: str
): # type: (...) -> None
    """
    rename onnx_graph value_name in onnx_graph{input, output, value_info, initializer}
    """
    for value_info in list(onnx_graph.input) \
            + list(onnx_graph.output) \
            + list(onnx_graph.value_info):
        if value_info.name == src_value_name:
            value_info.name = dst_value_name

    for tensor in onnx_graph.initializer:
        if tensor.name == src_value_name:
            node.name = dst_value_name


def value_info(
    onnx_graph, # type: onnx.GraphProto
    value_name # type: str
): # type: (...) -> dict
    """get about onnx_graph{input, output, value_info, initializer} by value_name"""
    value_info_book = {}

    input = filter(lambda x: x.name == value_name, onnx_graph.input)
    if input:
        value_info_book['input'] = input

    output = filter(lambda x: x.name == value_name, onnx_graph.output)
    if output:
        value_info_book['output'] = output

    value_info = filter(lambda x: x.name == value_name, onnx_graph.value_info)
    if value_info:
        value_info_book['value_info'] = value_info

    initializer = filter(lambda x: x.name == value_info, onnx_graph.initializer)
    if value_info:
        value_info_book['initializer'] = initializer

    return value_info_book
