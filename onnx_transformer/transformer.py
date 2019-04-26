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

import onnx
from onnx.helper import make_node
from proto import GraphProto, NodeProto
from . import helper


def required_transform(onnx_graph):
    """Conversion operations that must be performed"""
    break_self_loop_node(onnx_graph)


def break_self_loop_node(onnx_graph): # type: GraphProto -> None
    """
    Maybe has self-loop in fluid graph, like { value -> node, node -> value }
    break it to { value_0 -> node, node -> value_1 }, and update related nodes

    Effect:
        1. As result, no self-loop node in onnx_graph
        2. add some dup value when break self-loop nodes
    """
    # self_loop_nodes_book = { value_name_0: [ self_loop_node_0, self_loop_node_1 ], ... }
    self_loop_nodes_book = {}
    for node in onnx_graph.node:
        io_intersection = set(node.input) & set(node.output)

        # if node input & output has intersection, the node is self-loop node
        for value_name in io_intersection:
            if value_name not in self_loop_nodes_book:
                self_loop_nodes_book[value_name] = []
            self_loop_nodes_book[value_name].append(node)

    def dup_node_name(ori_node_name, dup_id):
        return '{}@dup_{}'.format(ori_node_name, dup_id)

    # break around value self-loop nodes one by one
    for v in self_loop_nodes_book:
        self_loop_nodes = self_loop_nodes_book[v]

        # update self-loop nodes
        for i, node in enumerate(self_loop_nodes):
            helper.rename_node_input(node, v, dup_node_name(v, i))
            helper.rename_node_output(node, v, dup_node_name(v, i + 1))

        # update value related nodes
        for node in onnx_graph.node:
            if node in self_loop_nodes:
                continue

            helper.rename_node_input(node, v, dup_node_name(v, len(self_loop_nodes)))
            helper.rename_node_output(node, v, dup_node_name(v, 0))

        # update rename value
        helper.rename_value(onnx_graph, v, dup_node_name(v, 0))


def add_split_op_for_shared_output(onnx_graph): # type: GraphProto -> None
    """
    output value shared with mulitple ops (as input) not legal in Anakin
    add split for shared output values
    """
    output_values = []
    for node in onnx_graph.node:
        for value_name in node.output:
            output_values.append(value_name)
    output_values = set(output_values)

    for value_name in output_values:
        has_this_input_nodes = filter(lambda x: value_name in x.input, onnx_graph.node)

        if len(has_this_input_nodes) <= 1:
            continue

        # output value shared with mulitple ops
        outputs = []

        # rename top nodes input to split_out
        for i, node in enumerate(has_this_input_nodes):
            split_out = 'split#{}#{}'.format(value_name, i)
            helper.rename_node_input(node, value_name, split_out)
            outputs.append(split_out)

        onnx_graph.node.extend([make_node(
            op_type='split',
            inputs=[value_name],
            outputs=outputs,
        )])
