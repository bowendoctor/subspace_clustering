#/usr/bin/env python
#! -*-coding:utf-8 -*-

#from time import time
import ConfigParser
import math
import os
import sys
import logging
import datetime
import traceback
import smtplib
import socket
from email.mime.text import MIMEText
from email.header import Header
from multiprocessing import Pool
from collections import defaultdict
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import StandardScaler

def div0(a, b):
    """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
    with np.errstate(divide='ignore', invalid='ignore'):
        c = np.true_divide(a, b)
        c[~ np.isfinite(c)] = 0  # -inf inf NaN
    return c


def wkmeans(k, data, oridata, init_labels, init_centers, beta, maxiter):
    try:
        n_samples, n_features = data.shape
        labels = init_labels
        centers = init_centers
        klist = []
        for ki in range(k):
            klist.append(ki)
        for it in range(maxiter):
            if len(klist) < 2:
                return -1, -1, -1, -1, -1
            newlabels = np.array([])
            weights = np.array([])
            j = 0
            while j < len(klist):
                i = klist[j]
                index = np.where(labels == i)
                if not index[0].any():
                    klist.remove(i)
                    continue
                ci = np.zeros(n_features)
                wi = np.zeros(n_features)
                ci = np.sum(np.power(data[index[0]] - centers[j], 2), axis=0)
                fea, cnts = np.unique(np.where(oridata[index[0]] == 0)[1], return_counts=True)
                ci[fea[np.where(cnts > 0.9 * data[index[0]].shape[0])]] = 0
                ci=np.nan_to_num(ci)
                ci[np.where(ci < 1e-8)] = 0
                for t in range(n_features):
                    tmp = np.sum(np.power(div0(ci[t], ci), 1 / (float(beta) - 1)))
                    wi[t] = 1 / tmp if tmp != 0 else 0
                if weights.shape[0] > 0:
                    weights = np.vstack((weights, wi))
                else:
                    weights = wi
                j += 1
            for i in range(n_samples):
                maxn = np.sum(np.power(data[i] - centers[klist[0]], 2) * weights[0])
                maxindex = 0
                for ci in range(1, len(klist)):
                    c = klist[ci]
                    tmpdist = np.sum(np.power(data[i] - centers[ci], 2) * weights[ci])
                    if tmpdist < maxn:
                        maxn = tmpdist
                        maxindex = c
                if newlabels.shape:
                    newlabels = np.hstack((newlabels, maxindex))
                else:
                    newlabels = np.array([maxindex])
            centers = np.mean(data[np.where(newlabels == klist[0])[0], :], axis=0)
            for i in range(1, len(klist)):
                centers = np.vstack((centers,
                                     np.mean(data[np.where(newlabels == klist[i])[0], :], axis=0)))
            labels = newlabels
        return 1, newlabels, weights, centers, klist
    except:
        return -1, -1, -1, -1, -1

def jaccard(a, b):
    maxret = 0
    for i in b:
        c = set(a) & set(i)
        ret = float(len(c)) / (len(a) + len(i) - len(c))
        if ret > maxret:
            maxret = ret
    return maxret


def newclassdiscover(k, it, batch, beta, maxiter, mu, std, uid, data_train,
                     labels_train, distances_train, threshold, importfea,
                     change_accept_rate, fea_accept_rate, threshold_rate, kernel_rate, email):
    try:
        n_samples = data_train.shape[0]
        index_outliers = []
        for i in range(n_samples):
            label = labels_train[i]
            if distances_train[i] > float(threshold[label]):
                index_outliers.append(i)
        uid_outliers = []
        for t in range(len(index_outliers)):
            uid_outliers.append(uid[index_outliers[t]])
        data_outliers = data_train[index_outliers]
        if data_outliers.shape[0] == 0:
            return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        data_outliers_std = (data_outliers - mu) / std

        labels_train_numpy = np.array(labels_train)
        labels_outliers = labels_train_numpy[index_outliers]

        bestflag, bestk, init_labels, init_centers = get_best_k(data_outliers_std, it, batch)
        if bestflag == -1:
            return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        flag, newlabels, newweights, newcenters, klist = wkmeans(bestk, data_outliers_std,\
        data_outliers, init_labels, init_centers, beta, maxiter)

        if flag == -1 or len(klist) < 2:
            return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        ret = []
        newfea = []
        for newk in klist:
            tmpindex = np.where(newlabels == newk)[0]
            maxn = 0
            for j in set(labels_outliers[tmpindex]):
                tmpj = int(np.sum(labels_outliers[tmpindex] == j))
                if maxn < tmpj:
                    maxn = tmpj
            if labels_outliers[tmpindex].shape[0] > 0 and float(maxn) / labels_outliers[tmpindex].shape[0] < change_accept_rate and len(tmpindex) > n_samples / len(klist):
                mu = np.mean(data_outliers_std[tmpindex, :], axis=0)
                var = np.var(data_outliers_std[tmpindex, :], axis=0)
                l = pickfeature(mu, var)
                rate = jaccard(l, importfea)
                if rate < fea_accept_rate:
                    ret.append(klist.index(newk))
                    newfea.append(l)
        if len(ret) < 1:
            return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1
        index = []
        for i in range(newlabels.shape[0]):
            if newlabels[i] in ret:
                index.append(i)
        retlabels = newlabels[index]
        retdata = data_outliers_std[index]
        retdist = []
        feadist = []
        for j in range(retdata.shape[0]):
            tmpclass = int(retlabels[j])
            tmpfeaindex = newfea[ret.index(tmpclass)]
            dist = math.sqrt(np.sum(np.power(retdata[j] - newcenters[int(retlabels[tmpclass])], 2) * (newweights[tmpclass] / np.sum(newweights[tmpclass]))))
            fdist = math.sqrt(np.sum(np.power(retdata[j][tmpfeaindex] - newcenters[int(retlabels[tmpclass])][tmpfeaindex], 2) * (newweights[tmpclass][tmpfeaindex] / np.sum(newweights[tmpclass][tmpfeaindex]))))
            retdist.append(dist)
            feadist.append(fdist)
        cnt = k
        for i in range(len(ret)):
            retlabels[retlabels == ret[i]] = cnt
            cnt += 1
        newthreshold = []
        newkerneldis = []
        newkernelimportdis = []
        for i in range(k, cnt):
            tmpindex = np.where(retlabels == i)[0].tolist()
            tmpdist = []
            for j in range(len(tmpindex)):
                tmpdist.append(retdist[j])
            sortdist = sorted(tmpdist, reverse=True)
            tmpfeadist = []
            for j in range(len(tmpindex)):
                tmpfeadist.append(feadist[j])
            sortfeadist = sorted(tmpfeadist, reverse=True)
            thresindex = int(len(sortdist)*threshold_rate)
            kernelindex = int(len(sortdist)*kernel_rate)
            kernelfeaindex = int(len(sortfeadist)*kernel_rate)
            newthreshold.append(sortdist[thresindex])
            newkerneldis.append(sortdist[kernelindex])
            newkernelimportdis.append(sortfeadist[kernelfeaindex])
        retuid = []
        for i in range(len(index)):
            retuid.append(uid_outliers[index[i]])
        return 1, newcenters[ret], newweights[ret], retlabels, retdist,\
        newfea, newthreshold, newkerneldis, newkernelimportdis, index
    except Exception, e:
        logging.info(traceback.format_exc())

if __name__ == '__main__':
    main()
