#!/usr/bin/python

import os, sys, os.path

file_dir = os.path.dirname(os.path.abspath(__file__))

weight_url = "http://cilvr.cs.nyu.edu/lib/exe/fetch.php?media=overfeat:overfeat-weights.tgz"

os.system("cd %s && mkdir -p data/default && cd data/default && wget %s -O weights.tgz && tar -xzf weights.tgz && rm weights.tgz"%(os.path.join(file_dir), weight_url))
