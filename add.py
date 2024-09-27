# Copyright 2023-2024 Synopsys, Inc.
# This Synopsys software and all associated documentation are proprietary
# to Synopsys, Inc. and may only be used pursuant to the terms and conditions
# of a written license agreement with Synopsys, Inc.
# All other use, reproduction, modification, or distribution of the Synopsys
# software or the associated documentation is strictly prohibited.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import networkx as nx

from .single_layer_transforms import remove_one_layer

""" Remove Identity node
"""


def RemoveIdentity(opt):
    G = opt.G
    TensorDict = opt.TensorDict
    shapeDict = opt.ShapeDict
    replace_dummpy_by_identity(opt, G, TensorDict, shapeDict)
    layers = list(nx.topological_sort(G))
    for layer in layers:
        op_type = G.nodes[layer].get("op_type", None)
        if op_type == "Identity":
            remove_one_layer(opt, layer)


def replace_dummpy_by_identity(opt, G, TensorDict, shapeDict):
    # Transpose: if only 1 dimension is not 1 (e.g. 1x1x1xC -> 1xCx1x1 is an identity)
    layers = list(nx.topological_sort(G))
    removed_nodes = []
    for layer in layers:
        if layer in removed_nodes:
            # if some dummy node has >1 outputs, then output[1:] will be removed from G,
            # skip them, otherwise Line 42 raises KeyError
            continue
        op_type = G.nodes[layer].get("op_type", None)
        # Remove Identity node that contain initializer
        if op_type == "Identity":
            if G.nodes[layer]["input"][0] not in TensorDict:
                continue
            identity_init_name = G.nodes[layer]["input"][0]
            TensorDict[layer] = TensorDict[identity_init_name]
            G.remove_node(layer)
            opt.passes_counter["RemoveIdentity"] += 1
        elif op_type == "Transpose":
            perm = G.nodes[layer]["attr_dict"].get("perm", None)
            increasing_order = [i for i in range(0, len(perm))]
            if perm == increasing_order:
                G.nodes[layer]["op_type"] = "Identity"
                G.nodes[layer]["attr_dict"] = {}
                continue

            out_shape = shapeDict.get(layer, [])
            if out_shape == []:
                continue
            non1 = [d for d in out_shape if d != 1]
            if len(non1) > 1:
                continue
            succs = list(G.successors(layer))
            flag = True
            for succ in succs:
                succ_op_type = G.nodes[succ].get("op_type", None)
                if succ_op_type != "Reshape":
                    flag = False
            if not flag:
                continue
            G.nodes[layer]["op_type"] = "Identity"
            G.nodes[layer]["attr_dict"] = {}
            opt.passes_counter["RemoveIdentity"] += 1
        elif op_type == "Dropout":
            # dropout has input ["data"] and attritube ["ratio"] and two outputs ["output", "mask"]
            # our converter assumes models in inference mode, so we just remove dropout nodes
            G.nodes[layer]["op_type"] = "Identity"
            G.nodes[layer]["attr_dict"] = {}
            if len(G.nodes[layer]["output"]) > 1:  # the 2nd output `mask` is optional
                _rm_out = G.nodes[layer]["output"].pop()
                G.remove_node(_rm_out)
                removed_nodes.append(_rm_out)
            opt.passes_counter["RemoveIdentity"] += 1
        elif op_type == "Add":
            # remove nop Add might affect the attention pattern
            inputs = G.nodes[layer]["input"]
            ip0 = TensorDict.get(inputs[0], None)
            ip1 = TensorDict.get(inputs[1], None)

            def _check_all_zeros_add(ip0, ip1, ip0_name, ip1_name):
                # if ip0 is None => both are none (i.e. not initializers)
                # if ip0 is not all-zeros => meaningful Add
                # condition len(ip0.shape) > 1, the Add node might be used for broadcasting
                # ip0: (1, 1, 1, 1) + ip1: (1) -> out: (1, 1, 1, 1)
                # remove Add node -> out (1)
                # if ip0 is None or ip0.any() or len(ip0.shape) > 1:
                if ip0 is None or ip0.any() or len(ip0.shape) > 1:
                    return
                # skip now, we can add this two constants and write back to initializer in the future
                if ip1 is not None and ip1.any():
                    return
                # assert ip1 is None, "Received Add(ip0, ip1) with both constant inputs."
                G.nodes[layer]["op_type"] = "Identity"
                G.nodes[layer]["attr_dict"] = {}
                # set the input as the non-initializer one
                G.nodes[layer]["input"] = [ip1_name]

            if ip0 is None:
                _check_all_zeros_add(ip1, ip0, inputs[1], inputs[0])
            else:
                _check_all_zeros_add(ip0, ip1, inputs[0], inputs[1])
            opt.passes_counter["RemoveIdentity"] += 1
        if op_type == "Resize":
            out_shape = shapeDict.get(layer, [])
            input0 = G.nodes[layer]["input"][0]
            if input0 in shapeDict:
                input_shape = shapeDict[input0]
                if input_shape == out_shape:
                    G.nodes[layer]["op_type"] = "Identity"
                    G.nodes[layer]["attr_dict"] = {}
                    # remove predecessor edges to avoid failure in remove_one_layer()
                    for pred in list(G.predecessors(layer)):
                        if pred != G.nodes[layer]["input"][0]:
                            G.remove_edge(pred, layer)
                    G.nodes[layer]["input"] = [G.nodes[layer]["input"][0]]
                    opt.passes_counter["RemoveIdentity"] += 1
