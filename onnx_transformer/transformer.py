import onnx
from proto import GraphProto, NodeProto

def break_self_loop(onnx_graph): # type: GraphProto -> None
    """There maybe has self-loop in fluid graph, like { value -> node, node -> value }
    break it to { value_0 -> node, node -> value_1 }, and update related nodes
    """
    # self_loop_nodes_book = { value_name_0: [ self_loop_node_0, self_loop_node_1 ], ... }
    self_loop_nodes_book = {}
    for node in onnx_graph.node:
        io_intersection = set(node.input) & set(node.output)

        # if node input & output has intersection, the node is self-loop
        for value_name in io_intersection:
            if value_name not in self_loop_nodes_book:
                self_loop_nodes_book[value_name] = []
            self_loop_nodes_book[value_name].append(node)

    # break around value self-loop nodes one by one
    for v in self_loop_nodes_book:
        self_loop_nodes = self_loop_nodes_book[v]

        # update self-loop nodes
        for i, node in enumerate(self_loop_nodes):
            node.input[:] = [x if x != v else '{}@{}'.format(v, i) for x in node.input]
            node.output[:] = [x if x != v else '{}@{}'.format(v, i + 1) for x in node.output]

        # update value related nodes
        for node in onnx_graph.node:
            if node in self_loop_nodes:
                continue

            node.input[:] = [x if x != v else '{}@{}'.format(v, len(self_loop_nodes)) for x in node.input]
            node.output[:] = [x if x != v else '{}@{}'.format(v, 0) for x in node.output]

        # update rename value
        for node in list(onnx_graph.input) \
                + list(onnx_graph.output) \
                + list(onnx_graph.value_info):
            if node.name == v:
                node.name = '{}@{}'.format(v, 0)

        for tensor in onnx_graph.initializer:
            if tensor.name == v:
                tensor.name = '{}@{}'.format(v, 0)


def transform(onnx_graph):
    break_self_loop(onnx_graph)
