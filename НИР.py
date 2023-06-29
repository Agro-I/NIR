#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install optuna')


# In[36]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.image import imread


import collections
import cv2
import os
import re
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.preprocessing import MinMaxScaler
import zipfile
import optuna


# In[4]:


path = "D:\Documents\Downloads\Total\Total\Blinks"
image_path_list = os.listdir(path)
image_names = [re.sub('m.png|.png', '', img) for img in image_path_list if 'm.png' not in img]
mask_names = [re.sub('m.png|.png', '', img) for img in image_path_list if 'm.png' in img]
intersection = set(image_names).intersection(mask_names)

uniq_img_names = list(set(intersection))
print(len(uniq_img_names), '- Количество пар фотографий/ масок')


# In[ ]:


def load_image(name, output=False):
    img = cv2.imread(path + '/' + name + '.png')
    return img

def load_mask(name, output=False):
    img_path = path + '\\' + name + 'm.png'
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    return np.array(img).reshape(img.shape[0], img.shape[1], 1)


def display(display_list, title_list=['Input Image', 'True Mask', 'Predicted Mask']):
    plt.figure(figsize=(15, 15))

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title_list[i])
        plt.imshow(display_list[i])
        plt.axis('off')
    plt.show()


# In[ ]:


def print_cluster(result, cluster_tag):
    result_copy = np.zeros_like(result)
    slice_img = np.where(result == cluster_tag, 1, 0)
    result_copy[..., :] = slice_img[..., np.newaxis]
    return result_copy


def get_IoU_for_slice(slice_, mask):
    intersection = np.logical_and(slice_, mask)
    union = np.logical_or(slice_, mask)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def get_IoU(slices):
    max_iou = np.max(slices)
    max_iou_index = np.argmax(slices)
    return max_iou, max_iou_index


def accuracy_IoU_scorer(result, mask):
    metrics_iou = []
    unique_tags = np.unique(result)
    for tag in unique_tags:
        result_copy = np.zeros_like(result)
        slice_img = np.where(result == tag, 1, 0)
        result_copy[..., :] = slice_img[..., np.newaxis]
        metrics_iou.append(get_IoU_for_slice(result_copy, mask))

    accuracy, cluster_ind = get_IoU(metrics_iou)
    slice_ = print_cluster(result, unique_tags[cluster_ind])
    return accuracy, slice_


# In[ ]:


def dbscan(data, eps=0.01, min_samples=6, metric='euclidean'):
    print(f"Начало выполнения DBSCAN с параметрами eps={eps}, min_samples={min_samples}")
    
    db = DBSCAN(eps=eps, min_samples=min_samples, metric=metric, n_jobs=-1).fit(data)
    labels = db.labels_

    n_clusters = len(np.unique(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print("Оцененное количество кластеров: %d" % n_clusters)
    print("Оцененное количество выбросов: %d" % n_noise)

    labeled_img = labels.reshape(200, 200)

    return labeled_img


# In[5]:


img = cv2.imread(path+'/1_Left_1.png')
mask = cv2.imread(path+'/1_Left_1m.png')
display([img,mask], ['Input Image','True Mask'])


# In[6]:


n = 10
images = np.array([load_image(img) for img in tqdm(uniq_img_names[:n])])
masks = np.array([load_mask(img) for img in tqdm(uniq_img_names[:n])])


# In[9]:


def objective(trial):
    eps = trial.suggest_uniform('eps', 2.5, 6.5)
    min_samples = trial.suggest_int('min_samples', 200, 500)

    db = dbscan(img_3f,eps=eps, min_samples=min_samples)

    # Calculate accuracy using accuracy_IoU_scorer function
    accuracy, _ = accuracy_IoU_scorer(db, mask)
    
    return accuracy

for i in range(n):
    img_3f = images[i].reshape((-1, 3))
    mask=masks[i]
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    best_trial = study.best_trial
    optimal_eps = best_trial.params['eps']
    optimal_min_samples = best_trial.params['min_samples']
    bestParamDBscan = dbscan(img_3f, optimal_eps, optimal_min_samples)
    accuracy, slice_ = accuracy_IoU_scorer(bestParamDBscan, masks[i])
    print(accuracy)
    directory = 'out/images/imagesDBScan/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open('out/images/imagesDBScan.txt', "a") as f:
        f.write(uniq_img_names[i] + ' ' + str(accuracy) + ' ' + str(optimal_eps)+ ' ' + str(optimal_min_samples) + '\n')
        f.close()

    data_2d = slice_.reshape((-1, slice_.shape[-1]))
    np.savetxt('out/images/imagesDBScan/' + (uniq_img_names[i]) + '.txt', data_2d, delimiter=',')


    


# In[21]:


def meanShift(data,bandwidth,cluster_all):
    print(f"Начало выполнения MeanShift с параметрами bandwidth={bandwidth}, cluster_all={cluster_all}")
    meanshift = MeanShift(bandwidth=bandwidth,cluster_all=True).fit(data)
    labels = meanshift.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print("Оцененное количество кластеров: %d" % n_clusters)
    print("Оцененное количество выбросов: %d" % n_noise)

    labeled_img = labels.reshape(200,200)
    return labeled_img


# In[32]:


def objective(trial):
    bandwidth = trial.suggest_int('bandwidth', 15, 25)
    cluster_all=trial.suggest_categorical('cluster_all', [True, False])

    db = meanShift(img_3f,bandwidth=bandwidth, cluster_all=cluster_all)

    # Calculate accuracy using accuracy_IoU_scorer function
    accuracy, _ = accuracy_IoU_scorer(db, mask)
    
    return accuracy

for i in range(n):
    img_3f = images[i].reshape((-1, 3))
    mask=masks[i]
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    best_trial = study.best_trial
    optimal_bandwidth = best_trial.params['bandwidth']
    optimal_cluster_all = best_trial.params['cluster_all']
    bestParamDBscan = meanshift(img_3f,bandwidth=optimal_bandwidth, cluster_all=optimal_cluster_all)
    accuracy, slice_ = accuracy_IoU_scorer(bestParamDBscan, masks[i])
    print(accuracy)
    directory = 'out/images/imagesMeanshift/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open('out/images/imagesMeanshift.txt', "a") as f:
        f.write(uniq_img_names[i] + ' ' + str(accuracy) + ' ' + str(optimal_cluster_all)+ ' ' + str(optimal_bandwidth) + '\n')
        f.close()

    data_2d = slice_.reshape((-1, slice_.shape[-1]))
    np.savetxt('out/images/imagesMeanshift/' + (uniq_img_names[i]) + '.txt', data_2d, delimiter=',')


# In[27]:


img_3f = images[1].reshape((-1, 3))
print(estimate_bandwidth(img_3f))
db = meanShift(img_3f,bandwidth= estimate_bandwidth(img_3f), cluster_all=True)
accuracy, slice_ = accuracy_IoU_scorer(bestParamDBscan, masks[i])
display([images[1], masks[1],db])


# In[31]:


accuracy, slice_ = accuracy_IoU_scorer(bestParamDBscan, masks[1])
print(accuracy)


# In[28]:


display([images[1], masks[1],db])


# In[47]:


def k_means(data,n_clusters,algorithm):
    print(f"Начало выполнения k_means с параметрами n_clusters={n_clusters}, algorithm={algorithm}")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, algorithm=algorithm, n_init=50).fit(data)
    labels = kmeans.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print("Оцененное количество кластеров: %d" % n_clusters)
    print("Оцененное количество выбросов: %d" % n_noise)

    labeled_img = labels.reshape(200,200)
    return labeled_img


# In[49]:


def objective(trial):
    n_clusters = trial.suggest_int('n_clusters', 10, 25)
    algorithm=trial.suggest_categorical('algorithm', ["lloyd", "elkan", "auto", "full"])

    db = k_means(img_3f,n_clusters=n_clusters, algorithm=algorithm)

    # Calculate accuracy using accuracy_IoU_scorer function
    accuracy, _ = accuracy_IoU_scorer(db, mask)
    
    return accuracy

for i in range(n):
    img_3f = images[i].reshape((-1, 3))
    mask=masks[i]
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    best_trial = study.best_trial
    n_clusters = best_trial.params['n_clusters']
    algorithm = best_trial.params['algorithm']
    bestParamDBscan = k_means(img_3f,n_clusters=n_clusters, algorithm=algorithm)
    accuracy, slice_ = accuracy_IoU_scorer(bestParamDBscan, masks[i])
    print(accuracy)
    directory = 'out/images/imagesKMeans/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open('out/images/imagesKMeans.txt', "a") as f:
        f.write(uniq_img_names[i] + ' ' + str(accuracy) + ' ' + str(n_clusters)+ ' ' + str(algorithm) + '\n')
        f.close()

    data_2d = slice_.reshape((-1, slice_.shape[-1]))
    np.savetxt('out/images/imagesKMeans/' + (uniq_img_names[i]) + '.txt', data_2d, delimiter=',')


# In[40]:


img_3f = images[1].reshape((-1, 3))
db = k_means(img_3f,n_clusters=10, algorithm="auto")
display([images[1], masks[1],db])
accuracy, slice_ = accuracy_IoU_scorer(bestParamDBscan, masks[1])
print(accuracy)


# In[ ]:




