
# coding: utf-8

# In[1]:


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '2' 
from __future__ import division, print_function
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

from tf_unet import image_util
from tf_unet import unet
from tf_unet import util


# In[2]:


net = unet.Unet(channels=3, n_class=2, layers=2, features_root=300)
trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2))


# In[5]:


data_provider = image_util.ImageDataProvider("/mnt/ccipd_data/CCF/ccfMaskTmp/training/*.png",                                  data_suffix='_img.png', mask_suffix='_mask.png')


# In[4]:


path = trainer.train(data_provider, "./unet_trained", training_iters=32, epochs=1, display_step=2)


# In[2]:


validation_provider = image_util.ImageDataProvider("/mnt/ccipd_data/CCF/ccfMaskTmp/test/*.png",                                  data_suffix='_img.png', mask_suffix='_mask.png')
x_test, y_test = validation_provider(1)
prediction = net.predict("./unet_trained/model.ckpt", x_test)

fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(12,5))

ax[0].imshow(np.squeeze(x_test,0), aspect="auto")
ax[1].imshow(y_test[0,...,1], aspect="auto")
mask = prediction[0,...,1] > 0.5
pro = prediction[0,...,1]
ax[2].imshow(mask, aspect="auto")
ax[3].imshow(pro, aspect="auto")
ax[0].set_title("Input")
ax[1].set_title("Ground truth")
ax[2].set_title("Prediction")
ax[3].set_title("Probability")
fig.tight_layout()
plt.figure()
plt.hist(prediction[0,...,1].ravel())

