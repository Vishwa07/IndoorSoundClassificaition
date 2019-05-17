# -*- coding: utf-8 -*-
"""
Created on Thu May 16 20:14:35 2019

@author: vmarimut
"""

arrays = np.random.randn(3, 4)
print(arrays)
print(np.stack(arrays, axis=-1))