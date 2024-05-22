#!/bin/bash

mkdir data
cd data
rm -rf dcs
rm -rf texts
rm -rf dataset
git clone https://github.com/ambuda-org/texts.git
git clone https://github.com/ambuda-org/dcs.git
cd ..
wget https://raw.githubusercontent.com/ambuda-org/gretil/main/1_sanskr/tei/sa_rAmAyaNa.xml -P data/texts/gretil
