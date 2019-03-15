#!/usr/bin/env python
# coding: utf-8

# In[2]:


def edaPlot(x:list,y:list,title:str,saveFile:bool):
    import numpy as np
    import matplotlib.pyplot as plt
    """x is the attribute, y is the average stars"""
    fig,ax=plt.subplots()
    ax.bar(x,y,facecolor='blue',width=0.4)
    ax.set_xlabel("factor level")
    ax.set_ylabel("average stars")
    ax.set_title(title)
    if saveFile:
        fig.savefig('./fig/{}'.format(title), format='png',dpi=800)

