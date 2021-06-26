# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 16:27:24 2021

@author: Mehak
"""

import setuptools
with open("README.md", "r") as fh:
    long_description = fh.read()
    
setuptools.setup(
     name='CPP',  
     version='1.0',
     scripts=['train_json.py'] ,

     install_requires=['keras==2.3', 'tensorflow==2.0',  'albumentations' , 'medpy' ],

     author="Mehak Arora",
     author_email="mehakarora@iisc.ac.in",
     description="A tool for Medical Image Segmentation",
     packages=setuptools.find_packages(),
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: MIT License",
         "Operating System :: OS Independent",
     ],
 )