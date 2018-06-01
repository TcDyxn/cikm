#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  1 00:11:47 2018

@author: tcd
"""

import mxnet as mx
import gluonnlp as nlp


seqen_1 = []
seqen_2 = []
seqsp_1 = []
seqsp_2 = []
label = []
with open('/home/tcd/train_dir/cikm/cikm_english_train_20180516.txt') as f:
    for i,line in enumerate(f.readlines()):
        line = line.split('\t')
        seqen_1.append(line[0])
        seqen_2.append(line[2])
        seqsp_1.append(line[1])
        seqsp_2.append(line[3])
        label.append(int(line[4][0]))
    f.close()

sp_unk = []
en_unk = []
with open('/home/tcd/train_dir/cikm/cikm_unlabel_spanish_train_20180516.txt') as f:
    for line in f.readlines():
        line = line.split('\t')
        sp_unk.append(line[0])
        en_unk.append(line[1])
    f.close()

counter = nlp.data.count_tokens(seqsp_1 + seqsp_2 + sp_unk)
sp_vocab = nlp.Vocab(counter)

counter = nlp.data.count_tokens(seqen_1 + seqen_2 + en_unk)
en_vocab = nlp.Vocab(counter)




















