#!/bin/bash

mkdir data
cd data
rm -rf data/dcs
rm -rf data/texts
git clone https://github.com/ambuda-org/texts.git
git clone https://github.com/ambuda-org/dcs.git
cd ..
wget https://raw.githubusercontent.com/ambuda-org/gretil/main/1_sanskr/tei/sa_rAmAyaNa.xml -P data/texts/gretil
