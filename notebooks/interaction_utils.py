"""
This file contains some common helper code for the analysis notebooks.
"""

# System
import os
import yaml
import pickle
from collections import namedtuple
from IPython.display import clear_output


# Externals
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch.nn as nn
# from torch_geometric.nn import GCNConv
from torch_geometric.nn import MessagePassing
import torch_geometric.nn as tnn
from torch_geometric.utils import add_self_loops, degree
from torch_scatter import *

from functools import partial

# Locals
from torch_geometric.data import Batch
from toy_utils import *

# Interactive
import ipywidgets as widgets
from ipywidgets import interact, interact_manual

#_________________________________ Dataset Generation _____________________

""" Generate the slider parameters """

def sliders(field, datasetYamlPath = "/global/u2/d/danieltm/ExaTrkX/eta-tracker/notebooks/datasetConfig.yaml"):
    with open(datasetYamlPath,"r") as f:
        datasetConfig = yaml.safe_load(f)
    kwargs = {k: widgets.FloatSlider(
            value=v['default'],
            min=v['min'],
            max=v['max'],
            step=0.1,
            description='%s' % k,
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        ) for k, v in datasetConfig[field].items()}
#     kwargs = {'w{}'.format(i):slider for i, slider in enumerate(weight_sliders)}
    return kwargs

#_______________________________ Architecture Generation _____________________

def return_same(**kwargs):
    i = []
    for k, v in kwargs.items():
        i.append(v)
    return i

def return_value(**kwargs):
    i = []
    for k, v in kwargs.items():
        i.append(v.value)
    return i

def dropdowns_range(field, datasetYamlPath = "/global/u2/d/danieltm/ExaTrkX/eta-tracker/notebooks/architectureConfig.yaml"):
    with open(datasetYamlPath,"r") as f:
        datasetConfig = yaml.safe_load(f)
    kwargs = {'%s_%s' % (k1, k2): widgets.Dropdown(
            options=list(range(i['min'], i['max'])),
            value=i['default'],
            description='%s of unique %s %s' % (field, k1, k2),
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        ) for k1, v in datasetConfig[field].items() for k2, i in v.items()}
    return kwargs


def dropdown_list(field, subfield, subsubfield, index, datasetYamlPath = "/global/u2/d/danieltm/ExaTrkX/eta-tracker/notebooks/architectureConfig.yaml"):
    with open(datasetYamlPath,"r") as f:
        datasetConfig = yaml.safe_load(f)
    newDropdown = widgets.Dropdown(
            options=datasetConfig[field][subfield][subsubfield],
            description='%s of %s %s %s ' % (field, subfield, index, subsubfield),
            disabled=False,
            continuous_update=False,
            orientation='horizontal',
            readout=True,
            readout_format='.1f',
        )
    return newDropdown

def dropdown_watcher(gen_1, gen_2, n_e_g, c_p, layer_list):
    dropdown_partial = partial(dropdown_maker, gen_2, n_e_g, c_p, layer_list)
    gen_1.observe(dropdown_partial, 'value')
    
def dropdown_maker(gen_2, n_e_g, c_p, layer_list, *args):
    new_dropdowns = [dropdown_list("methods", n_e_g, c_p, i) for i in range(args[0].new)]
    kwargs = {'w{}'.format(i):drop for i, drop in enumerate(new_dropdowns)}
    gen_2.children = tuple(new_dropdowns)
    layer_list.update({k.description: k.value for k in gen_2.children})
    set_result_partial = partial(set_result, layer_list)
    [child.observe(set_result_partial, 'value') for child in gen_2.children]
    
# def build_layer_list(layer_generators, b):
# #     layer_list = [j.value for i in layer_generators for j in i.children]
#     print([try i.children if  for i in layer_generators])
# #     print(layer_list)
#     return layer_list

def set_result(layer_list, *args):
#     gen.result = [child.value for child in gen.children]
#     print(args[0])
#     print("SOMETHING CHANGED")
    layer_list[args[0].owner.description] = args[0].new
