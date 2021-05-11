
# coding: utf-8

# In[8]:


import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = '2' 
from __future__ import division, print_function
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from tf_unet import image_util
from tf_unet import unet
from tf_unet import util
print(image_util.__file__)


# In[9]:


import scipy.misc

def savingPrediction(predictions, fileNames):
    for i in range(len(fileNames)):

        baseName = fileNames[i]
        baseName = baseName.replace('/mnt/ccipd_data/CCF/lungMsk', '/mnt/data/home/xxw345/Data/tumorPredictedMask')
#        print(type(baseName))
        prob0 = baseName.replace('.png','_mask0.png')
        prob1 = baseName.replace('.png','_mask1.png')
#        print(prob0)
#        print(prob1)
        scipy.misc.imsave(prob0, predictions[i,:,:,0])
        scipy.misc.imsave(prob1, predictions[i,:,:,1])


# In[10]:


### batch processing for all images under folder
# net = unet.Unet(channels=3, n_class=2, layers=2, features_root=300)
# validation_provider = image_util.PredictionImageDataProvider("/mnt/ccipd_data/CCF/lungMsk/TCGA_LUAD/*.png",\
#                                   data_suffix='.png')

# for i in range(0,len(validation_provider.data_files),10):
#     x_prediction = validation_provider(10)    
#     y_prediction = net.predict("./unet_trained/model.ckpt", x_prediction)
#     savingPrediction(y_prediction, validation_provider.data_files[i:min(i+x_prediction.shape[0], len(validation_provider.data_files))])    
    


# In[25]:


#test prediction for visualizing the results
validation_provider = image_util.PredictionImageDataProvider("/mnt/ccipd_data/CCF/lungMsk/TCGA_LUAD/*.png",                                  data_suffix='.png')
x_prediction = validation_provider(1)
y_prediction = net.predict("./unet_trained/model.ckpt", x_prediction)

fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(12,5))
ax[0].imshow(np.squeeze(x_prediction,0), aspect="auto")
mask = y_prediction[0,...,1] > 0.18640
pro = y_prediction[0,...,1]
ax[1].imshow(pro, aspect="auto")
ax[2].imshow(mask, aspect="auto")
ax[0].set_title("Input")
ax[1].set_title("Probability")
ax[2].set_title("Binary")
fig.tight_layout()
plt.figure()
plt.hist(y_prediction[0,...,1].ravel())

