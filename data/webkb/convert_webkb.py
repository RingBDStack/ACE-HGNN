import os
import pickle as pkl
import sys
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
from utils.gat_utils import encode_onehot
name="cornell"
file = open(name+".cites")
web_dict={}
count=0
for line in file.readlines():
  print(line)
  source,target=line.split(" ")[0],line.split(" ")[1]
  source_w,target_w=0,0
  if source in web_dict:
    source_w=web_dict[source]
  else:
    web_dict[source]=str(count)
    source_w=str(count)
    count=count+1
  if target in web_dict:
    target_w=web_dict[source]
  else:
    web_dict[target]=str(count)
    target_w=str(count)
    count=count+1
  with open(name+'1.cites', 'a', encoding='utf-8', newline='') as f:
    f.write(source_w+" "+target_w+"\n")


