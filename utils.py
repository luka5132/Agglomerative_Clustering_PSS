from scipy.spatial import distance
from sklearn.cluster import AgglomerativeClustering
from metricutils import *
import numpy as np
import os


def page_con_matrix(n_pages, window = 1):
  #function to create the connectivity matrix for a stream
  #creates (n,n) matrix where the diagonal+1 and diagonal -1 are all 1 and the
  #rest of the cells are 0
  empty_matrix = np.zeros((n_pages,n_pages))
  for n in range(1,window+1):
    for i in range(n_pages):
      upper_diag = min(n_pages-1,i+n)
      if not i == upper_diag:
        empty_matrix[i][upper_diag] =1
      lower_diag = max(0,i-n)
      if not i == lower_diag:
        empty_matrix[i][lower_diag] =1
  return empty_matrix

def groups_to_lengths(aggl_preds):
  ### function to turn predictions from aggl clustering into the
  ### answer format used in reserach, namely a list with lengths of papers
  first = False
  tot_preds = len(aggl_preds) -1
  current_group = aggl_preds[0]
  current_length = 0
  outputs = []
  for i,pred in enumerate(aggl_preds):

    if pred == current_group:
      current_length += 1
    else:
      current_group = pred
      outputs.append(current_length)
      current_length = 1

    if i == tot_preds:
      outputs.append(current_length)
  return outputs

def cluster_with_switch(labels,vectors, switch_first = True, labels_bin = False, mult = 2):
  if not labels_bin:
    bin_labels = length_list_to_bin(labels)
  else:
    bin_labels = labels
  n_pages = len(bin_labels)
  n_docs = int(sum(bin_labels))
  c_mat = page_con_matrix(n_pages) 

  dist_list = []
  for i in range(len(vectors)-1):
    current_vector = vectors[i]
    next_vector = vectors[i+1]
    dist = distance.cosine(current_vector, next_vector)
    dist_list.append(dist)
  dist_list = np.array(dist_list)
  if len(dist_list) >1:
    dist_list_norm = (dist_list - np.min(dist_list)) / (np.max(dist_list) - np.min(dist_list))
    nth_highest = np.sort(dist_list_norm)[-n_docs]
  else:
    dist_list_norm = dist_list
    nth_highest = dist_list[0]
  
  avg = np.mean(dist_list_norm)
  dist_mat = np.zeros((n_pages,n_pages))
  N = len(dist_list_norm)
  if switch_first:
    dist_list_norm[0] = max(mult*avg-dist_list_norm[0],0)
  for i,dist in enumerate(dist_list_norm):
    if i < N-1:
      if dist > nth_highest:
        next_dist = dist_list_norm[i+1]
        dist_list_norm[i+1] = max(mult*avg-next_dist,0)

  for i,dist in enumerate(dist_list_norm):
    if i < n_pages:
      dist_mat[i,i+1] = dist
      dist_mat[i+1,i] = dist


  cluster = AgglomerativeClustering(n_clusters=n_docs, affinity='precomputed', linkage='average',compute_distances = True, connectivity=c_mat)  
  return dist_list_norm, length_list_to_bin(groups_to_lengths(cluster.fit_predict(dist_mat)))

def select_topn(pred_list, n):
  pred_labels = np.zeros(len(pred_list))
  inds = np.argpartition(pred_list, -n)[-n:]
  for ind in inds:
    pred_labels[ind] = 1
  return pred_labels

def sum_metrics(stream_results, metric_list = ['Boundary','Bcubed','WindowDiff','Block','Weighted Block']):
  f1_res = stream_results['F1']
  tot_metric_score = 0
  for met,val in f1_res.items():
    if met in metric_list:
      tot_metric_score += val
  return tot_metric_score

def get_base_metrics(gs, pred):
  tot_1 = sum(gs)
  comb = pred - gs
  fn = len(np.where(comb == -1)[0])
  fp = len(np.where(comb == 1)[0])
  tp = tot_1 - fn
  tn = len(gs) - tot_1 - fn
  return tp,tn,fp,fn	

def vec_comparrison(vector, vector_list):
  if len(vector_list) > 0:
    tot_dist = 0
    for vec in vector_list:
      if (vector == vec).all():
        continue
      tot_dist += distance.cosine(vector, vec)
    return tot_dist / len(vector_list)
  else:
    return 0

def distance_against_eachother(vector_list, gold_standard):
  gs_len = bin_to_length_list(gold_standard)
  longer_than_two = len([x for x in gs_len if x > 1])
  index = 0
  results= {'same_document':0,'different_document':0}
  for doc in gs_len:
    this_doc = index+doc
    if doc < 2:
      index = this_doc
      continue
    same_doc = 0
    diff_doc = 0
    vectors_before = vector_list[:index]
    current_vectors = vector_list[index:this_doc]
    vectors_after = vector_list[this_doc:]
    for i, vec in enumerate(current_vectors):
      same = vec_comparrison(vec,current_vectors[i+1:])
      bef = vec_comparrison(vec,vectors_before)
      aft = vec_comparrison(vec,vectors_after)
      same_doc += same
      diff_doc += (bef+aft) /2
    
    index = this_doc
    results['same_document'] += same_doc / (doc - 1)
    results['different_document'] += diff_doc / doc

    index = this_doc
  results['same_document'] = (results['same_document'] / longer_than_two)
  results['different_document'] = (results['different_document'] / longer_than_two)

  return results

def round_threshold(pred_arr, t):
  is_higher = pred_arr > t
  return is_higher.astype(np.int_)
  