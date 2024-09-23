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
import numpy as np

from nnac.core.log import Logger

from .single_layer_transforms import remove_one_layer, append_new_node

logger = Logger("OPTIMIZATION")

"""
Fuse Group Normalization
"""

def FuseGroupNormalization(opt):
    G = opt.G
    tensorDict = opt.TensorDict

    layers = list(nx.topological_sort(G))
    for layer in layers:
        if layer in G.nodes and G.nodes[layer].get("op_type", None) == "Reshape":
          succs = list(G.successors(layer))
          if len(succs) != 1 or G.nodes[succs[0]].get("op_type", None) != "InstanceNormalization":
            continue
          
          instance_norm_layer = succs[0]
          succs_instance_norm = list(G.successors(instance_norm_layer))
          if len(succs_instance_norm) != 1 or G.nodes[succs_instance_norm[0]].get("op_type", None) != "Reshape":
            continue
          
          second_reshape_layer = succs_instance_norm[0]
          succs_second_reshape = list(G.successors(second_reshape_layer))
          if len(succs_second_reshape) != 1 or G.nodes[succs_second_reshape[0]].get("op_type", None) != "Mul":
            continue
          
          mul_layer = succs_second_reshape[0]
          succs_mul = list(G.successors(mul_layer))
          if len(succs_mul) != 1 or G.nodes[succs_mul[0]].get("op_type", None) != "Add":
            continue
          
          add_layer = succs_mul[0]

          scale = tensorDict.get(G.nodes[mul_layer]["input"][1], None)
          bias = tensorDict.get(G.nodes[add_layer]["input"][1], None)
          if scale is None or bias is None:
            continue
          
          print(instance_norm_layer)
          num_channels = G.nodes[instance_norm_layer][0].shape[1]
          print(num_channels)
          exit()
          num_groups = num_channels

          # group_norm_node = {
          #   "op_type": "GroupNormalization",
          #   "attr_dict": {
          #     "num_groups": num_groups,
          #     "epsilon": G.nodes[instance_norm]["attr_dict"].get("epsilon", 1e-5),
          #     "scale": scale,
          #     "bias": bias,
          #   },
          #   "input":G.nodes[first_reshape]["input"],
          #   "output": G.nodes[add_layer]["output"]
          # }

          # G.add_node(f"group_norm_{layer}", **group_norm_node)
          # G =  nx.relabel_nodes(G, {add_layer: f"group_norm_{layer}"})
          # G.remove_node(first_reshape)
          # G.remove_node(instance_norm)
          # G.remove_node(second_reshape)
          # G.remove_node(mul_layer)
          # G.remove(add_layer)

          # opt.passes_counter["fuse_group_norm"] += 1

          new_node_name = layer + "GroupNormalization"
          new_node_input = G.nodes[first_reshape]["input"]
          new_node_output = G.nodes[add_layer]["output"]
          append_new_node(
              opt,
              layer, new_node_name,
              "GroupNormalization",
              [new_node_input],
              [new_node_name],
              {"num_groups": num_groups, "epsilon": G.nodes[instance_norm]["attr_dict"].get("epsilon", 1e-5), "scale": scale, "bias": bias}
          )

          G.remove_node(first_reshape)
          G.remove_node(instance_norm)
          G.remove_node(second_reshape)
          G.remove_node(mul_layer)
          G.remove(add_layer)

          opt.passes_counter["FuseGroupNormalization"] += 1


