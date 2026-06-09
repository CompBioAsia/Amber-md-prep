#!/bin/sh
git clone https://github.com/CompBioAsia/Amber-md-prep
pip install numpy==2.2.6
pip install mdtraj==1.11.1.post1
pip install git+https://github.com/CharlieLaughton/Alphafix.git
pip install git+https://github.com/CompBioAsia/CBAtools.git
