import numpy as np
import torch

def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH

def calc_map(qB, rB, queryL, retrievalL, knn):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    p = 0
    map = 0
    r_2 = 0.0
    #print(num_query)
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        #print(iter)
        #gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.int64)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        #print(hamm)
        true_in_r_2 = (hamm <= 2)
        #if np.where(true_in_r_2 == True)[0].shape[0] != 0:
        #print(np.sum(true_in_r_2))
        if np.sum(true_in_r_2) !=0:
            r_2_ = np.sum(true_in_r_2 * gnd) / np.sum(true_in_r_2)
        else:
            r_2_ = 0.0
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        p_ = np.sum(gnd[:knn] / knn)
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        map = map + map_
        p = p + p_
        r_2 = r_2 + r_2_
    map = map / num_query
    p = p / num_query
    #print(r_2)
    r_2 = r_2/ num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return map, p, r_2


def calc_map_all(qB, rB, queryL, retrievalL, knn,device):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]

    # rB = torch.from_numpy(rB).to(device)
    # retrievalL = torch.from_numpy(retrievalL).to(device)
    # qB = torch.from_numpy(qB).to(device)
    # queryL = torch.from_numpy(queryL).to(device)
    
    # p_all = torch.zeros(num_query)
    # map_all = torch.zeros(num_query)
    # r_2_all = torch.zeros(num_query)

    p_all = np.zeros(num_query)
    map_all = np.zeros(num_query)
    r_2_all = np.zeros(num_query)
    code_length = rB.shape[1]
    for iter in range(num_query):
        # start = time.time()
        gnd = (queryL[iter, :] @ retrievalL.t()) > 0.
        tsum = gnd.sum().item()
        if tsum == 0:
            continue
        hamm = 0.5 * (code_length - qB[iter, :] @ rB.t())

        
        true_in_r_2 = (hamm <= 2)
        
        if true_in_r_2.sum().item() !=0:
            r_2_ = (true_in_r_2 * gnd).sum() / (true_in_r_2).sum()
        else:
            r_2_ = torch.zeros(1)

        ind = torch.argsort(hamm)
        gnd = gnd[ind]
        p_ = torch.sum(gnd[:knn] / knn)
        count = torch.linspace(1, tsum, tsum).to(device)
        

        
        tindex = (torch.nonzero(gnd == 1, as_tuple=False).squeeze() + 1.0).float()
        map_ = (count / tindex).mean()

        map_all[iter] = map_.item()
        p_all[iter] = p_.item()
        r_2_all[iter] = r_2_.item() 
        # end = time.time()
        # print(end-start) 
    torch.cuda.empty_cache() 
    return map_all, p_all, r_2_all

def shot_eval(preds, labels, train_data, many_shot_thr=100, low_shot_thr=20):
    if isinstance(train_data, np.ndarray):
        training_labels = np.array(train_data).astype(int)
    else:
        training_labels = np.array(train_data.dataset.labels).astype(int)

    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))

    
    train_class_count = []
    query_class_count = []
    class_sum = []
    for l in np.unique(labels):
        train_class_count.append(len(training_labels[training_labels == l]))
        query_class_count.append(len(labels[labels == l]))
        class_sum.append(preds[labels == l].sum())

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(len(train_class_count)):
        if train_class_count[i] > many_shot_thr:
            many_shot.append((class_sum[i] / query_class_count[i]))
        elif train_class_count[i] < low_shot_thr:
            low_shot.append((class_sum[i] / query_class_count[i]))
        else:
            median_shot.append((class_sum[i] / query_class_count[i]))    
 
    if len(many_shot) == 0:
        many_shot.append(0)
    if len(median_shot) == 0:
        median_shot.append(0)
    if len(low_shot) == 0:
        low_shot.append(0)

    return np.mean(many_shot), np.mean(median_shot), np.mean(low_shot)


def calc_map_tde(qB, rB,q_cls_spec,r_cls_spec, queryL, retrievalL, knn,alpha):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    p = 0
    map = 0
    r_2 = 0.0
    #print(num_query)
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    for iter in range(num_query):
        #print(iter)
        #gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.int64)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm_code = calc_hammingDist(qB[iter, :], rB)
        hamm_cls = calc_hammingDist(q_cls_spec[iter, :], r_cls_spec)
        hamm = hamm_code - alpha * hamm_cls
        true_in_r_2 = (hamm <= 2)
        if np.sum(true_in_r_2) !=0:
            r_2_ = np.sum(true_in_r_2 * gnd) / np.sum(true_in_r_2)
        else:
            r_2_ = 0.0
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        p_ = np.sum(gnd[:knn] / knn)
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        
        map = map + map_
        p = p + p_
        r_2 = r_2 + r_2_


    map = map / num_query
    p = p / num_query
    r_2 = r_2/ num_query
    
    
    return map, p, r_2



def calc_map_per_cls(qB, rB, queryL, retrievalL, knn):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    p = np.zeros(num_query)
    map = np.zeros(num_query)
    r_2 = np.zeros(num_query)

    for iter in range(num_query):
        #print(iter)
        #gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.int64)
        tsum = np.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        #print(hamm)
        true_in_r_2 = (hamm <= 2)
        #if np.where(true_in_r_2 == True)[0].shape[0] != 0:
        #print(np.sum(true_in_r_2))
        if np.sum(true_in_r_2) !=0:
            r_2_ = np.sum(true_in_r_2 * gnd) / np.sum(true_in_r_2)
        else:
            r_2_ = 0.0
        ind = np.argsort(hamm)
        gnd = gnd[ind]
        p_ = np.sum(gnd[:knn] / knn)
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(gnd == 1)) + 1.0
        map_ = np.mean(count / (tindex))
        # print(map_)
        # map = map + map_
        # p = p + p_
        # r_2 = r_2 + r_2_
        map[iter] = map_
        p[iter] = p_
        r_2[iter] = r_2_
    queryL_new = np.argmax(queryL,axis=1)
    classes = np.unique(queryL_new)
    map_per_cls = []
    p_per_cls = []
    r_2_per_cls = []
    for the_class in classes:
        idx = np.where(queryL_new == the_class)[0]
        cur_num = idx.shape[0]
        cur_map = map[idx]
        cur_p = p[idx]
        cur_r_2 = r_2[idx]
        map_per_cls.append(cur_map.sum() / cur_num)
        p_per_cls.append(cur_p.sum() / cur_num)
        r_2_per_cls.append(cur_r_2.sum() / cur_num)
    return map_per_cls, p_per_cls, r_2_per_cls

def calc_topMap(qB, rB, queryL, retrievalL, topk):
    # qB: {-1,+1}^{mxq}
    # rB: {-1,+1}^{nxq}
    # queryL: {0,1}^{mxl}
    # retrievalL: {0,1}^{nxl}
    num_query = queryL.shape[0]
    topkmap = 0
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    for iter in range(num_query):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.int64)
        hamm = calc_hammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        # print(topkmap_)
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    # print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    return topkmap

if __name__=='__main__':
    pass