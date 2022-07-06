import numpy as np
import os
import itertools

def compute_mof(gt_list, pred_list):
    match, total = 0, 0
    match_bg, total_bg = 0, 0

    for gt_label, recognized in zip(gt_list, pred_list):
        gt_label = np.array(gt_label)
        recognized = np.array(recognized)
        correct = recognized == gt_label
        
        match += correct.sum()
        total += len(gt_label)
        
        length = len(gt_label)
        for i in range(length):
            if gt_label[i] != 0:
                total_bg += 1
                if gt_label[i] == recognized[i]:
                    match_bg += 1

    mof = match / total
    mof_bg = match_bg / total_bg
    return mof, mof_bg

def compute_IoU_IoD(gt_list, pred_list):

    IOU = []
    IOD = []
    for ground_truth, recognized in zip(gt_list, pred_list):

        ground_truth = np.array(ground_truth)
        recognized = np.array(recognized)
        unique = list(set(np.unique(ground_truth))) #.union(set(np.unique(recognized))) 

        video_iou = []
        video_iod = []
        for i in unique:
            recog_mask = recognized == i
            gt_mask = ground_truth == i
            union = np.logical_or(recog_mask, gt_mask).sum()
            intersect = np.logical_and(recog_mask, gt_mask).sum() # num of correct prediction
            num_recog = recog_mask.sum()
            
            video_iou.append(intersect / (union+ 1e-6))
            video_iod.append(intersect / (num_recog + 1e-6))

        
        IOU.append(np.mean(video_iou))
        IOD.append(np.mean(video_iod))
        
    return np.mean(IOU), np.mean(IOD)


def compute_score(gt_list, pred_list):
    '''
    compute scores depending on the pred_list.
    '''
    mof = compute_mof(gt_list, pred_list)
    iox = compute_IoU_IoD(gt_list, pred_list)
    result = {
        'MoF' : mof[0],
        'Mof-bg': mof[1],
        'IoU' : iox[0],
        'IoD' : iox[1],
    }
    return result


def decode(vfname, sequence, transcript, gt_label, OT_matrix, index2label, test_save_dir, args):
    '''
    transfer OT_matrix to predict_labels and save generated results.
    '''
    OT_matrix = OT_matrix.cpu().numpy()
    if args.enable_spectral:
        pred_labels = np.argmax(OT_matrix, axis=1)
    else:
        pred_labels = np.argmin(OT_matrix, axis=1)

    I = sequence.shape[1]
    I_ = OT_matrix.shape[0]

    # filter the predicted too small frame labels, and replace them with side frame labels
    segment_sum = 0
    segment_num = []
    segment_name = []

    i = 0
    while i < I_:
        label_type = pred_labels[i]
        label_count = 0
        while i < I_ and pred_labels[i] == label_type:
            label_count += 1
            i += 1

        segment_num.append(label_count)
        segment_name.append(label_type)
        segment_sum += 1

    for i in range(segment_sum):
        if segment_num[i] <= 2:
            if i == 0 or i == segment_sum - 1:
                continue
            else:
                if segment_num[i-1] >= segment_num[i+1]:
                    segment_num[i-1] += segment_num[i]
                    segment_num[i] = 0
                else:
                    segment_num[i+1] += segment_num[i]
                    segment_num[i] = 0

    i = 0
    for j in range(segment_sum):
        for k in range(segment_num[j]):
            pred_labels[i] = segment_name[j]
            i += 1
    
    pred_labels_ = np.zeros(I)
    # restore the complete frames from sampled frames
    sample_rate = args.sample_rate
    for i in range(I_):
        pred_labels_[i * sample_rate : min((i + 1) * sample_rate, I)] = transcript[pred_labels[i]]

    save_fname = os.path.join(test_save_dir, vfname + "_action_alignment")
    
    recog_labels = list(pred_labels)
    recog_labels = [transcript[k] for k, g in itertools.groupby(recog_labels)]

    with open(save_fname, 'w') as f:
        f.write( '### Recognized sequence: ###\n' )
        f.write( ' '.join( [index2label[s] for s in recog_labels] ) + '\n' )
        # f.write( '### Score: ###\n' + str(score) + '\n')
        f.write( '### Frame level recognition: ###\n')
        f.write( ' '.join( [index2label[l] for l in pred_labels_] ) + '\n' )

    # return predicted frame-level labels 
    return pred_labels_


