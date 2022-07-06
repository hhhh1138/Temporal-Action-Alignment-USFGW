import torch
from torch import optim
import numpy as np
import random
from .model import loss_function
from .ae_utils import key_frame, create_tools, ufgw_discrepancy
from .ae_utils import transfer_data_to_torch_mlp
from .ae_utils import k_pae, k_dae
import sys
sys.path.append('src')
from inference import compute_score, decode
from pprint import pprint
import os
import time
import json
from tqdm import tqdm
import matplotlib.pyplot as plt


def train(model_v, model_w, vfname, sequence, transcript, gt_label, index2label, device, optimizer, epoch, args, length_model, buffer, recloss, regloss_po, regloss_ne, totalloss):
    model_v.train()
    model_w.train()
    optimizer.zero_grad()

    data_v, data_w, data_w_ne = transfer_data_to_torch_mlp(sequence, transcript, index2label, args.xw_dim, args.sample_rate)
    data_v = data_v.to(device) # [I,64]
    data_w = data_w.to(device) # [J,48] if we take 'breakfast' as an example
    data_w_ne = data_w_ne.to(device) # [J_ne,48], J_ne is set by ourselves
    print('I, J, J_complement = {}, {}, {}'.format(data_v.shape[0], data_w.shape[0], data_w_ne.shape[0]))
    
    # generate distance matrices for vv, ww, vw
    if args.model_type == 'probabilistic':
        recon_batch_v, z_v, mu_v, logvar_v = model_v(data_v) 
        # sample mu and logvar
        mu_v = key_frame(mu_v, args.sample_rate, device) 
        logvar_v = key_frame(logvar_v, args.sample_rate, device) 
        recon_batch_w, z_w, mu_w, logvar_w = model_w(data_w) 
        recon_batch_w_ne, z_w_ne, mu_w_ne, logvar_w_ne = model_w(data_w_ne)# negative words set， Perform the same encoder operation
        K_v, K_w, K_vw, K_v_ne, K_w_ne, K_vw_ne = k_pae(mu_v, mu_w, mu_w_ne, logvar_v, logvar_w, logvar_w_ne, args, device)
    else:
        recon_batch_v, z_v = model_v(data_v)
        # sample z
        z_v = key_frame(z_v, args.sample_rate, device) 
        recon_batch_w, z_w = model_w(data_w)  
        recon_batch_w_ne, z_w_ne = model_w(data_w_ne)# negative words set， Perform the same encoder operation
        K_v, K_w, K_vw, K_v_ne, K_w_ne, K_vw_ne = k_dae(z_v, z_w, z_w_ne, args, device)

    rec_loss_v = model_v.rec_loss
    # rec_loss_v = loss_function(recon_batch_v, data_v, args.loss_type) # size of recon_batch_v: [I,64], FV feature dimension is 64
    rec_loss_w = loss_function(recon_batch_w, data_w, args.loss_type, transcript, device) # recon_batch_w: [J,48]
    rec_loss = rec_loss_v + rec_loss_w # rec loss

    print('rec_loss_v:{}'.format(rec_loss_v))
    print('rec_loss_w:{}'.format(rec_loss_w))
    print('rec_loss:{}'.format(rec_loss.item()))

    # compute action lengths distribution as prior
    length_distribution_p = []
    for item in transcript:
        times = np.sum(transcript == item)
        length_distribution_p.append(length_model.mean_lengths[item] / times)
    
    length_distribution_p = np.array(length_distribution_p, dtype=np.float32)
    length_distribution_p /= np.sum(length_distribution_p)
    length_distribution_p = torch.from_numpy(length_distribution_p).to(device)
    length_distribution_p = length_distribution_p.view(len(transcript), 1)

    reg_loss_po, T = ufgw_discrepancy(K_v, K_w, K_vw, device, args, length_distribution_p) # positive reg loss
    print('reg_loss_po:{}'.format(reg_loss_po.item()))
    if args.enable_contrastive_learning:
        reg_loss_ne, _ = ufgw_discrepancy(K_v_ne, K_w_ne, K_vw_ne, device, args) # negative reg loss
        print('reg_loss_ne:{}'.format(reg_loss_ne.item()))
        regloss_ne.append(reg_loss_ne.item())
        reg_loss = args.gamma * (reg_loss_po - reg_loss_ne) # reg loss
    else:
        reg_loss = args.gamma * (reg_loss_po) # reg loss
    
    print('reg_loss:{}'.format(reg_loss.item()))
    # get frame-level labels by decoding computed OT_matrix
    pred_labels = torch.argmax(T, axis=1)
    I = sequence.shape[1]
    I_ = T.shape[0]
    labels = torch.zeros(I).to(device)
    for i in range(I_):
        left = i * args.sample_rate
        right = min((i + 1) * args.sample_rate, I)
        labels[left : right] = transcript[pred_labels[i]]

    labels = labels.detach().cpu().numpy()

    loss = rec_loss + reg_loss
    loss.backward()
    optimizer.step()
    recloss.append(rec_loss.item())
    regloss_po.append(reg_loss_po.item())
    totalloss.append(loss.item())
    if epoch % args.print_every == 0:
        print('====> Iteration: {} Average RecLoss: {:.4f} RegLoss: {:.4f} TotalLoss: {:.4f}'.format(
        epoch, rec_loss.item(), reg_loss.item(), loss.item()))
    
    # during training, update buffer and length model
    buffer.add_sequence(vfname, sequence, transcript, labels, gt_label)
    length_model.update_mean_lengths(buffer)
    

def predict(model_v, model_w, data_v, data_w, data_w_ne, device, epoch, args, g_mat, length_distribution_p, transcript):
    '''
    testing phase, compute optimal transport matrix
    '''
    model_v.eval()
    model_w.eval()

    data_v = data_v.to(device) # [I,64]
    data_w = data_w.to(device) # [J,48] if we take 'breakfast' as an example
    data_w_ne = data_w_ne.to(device) # [J_ne,48], J_ne is set by ourselves

    if args.model_type == 'probabilistic':
        recon_batch_v, z_v, mu_v, logvar_v = model_v(data_v) 
        # sample mu and logvar
        mu_v = key_frame(mu_v, args.sample_rate, device) 
        logvar_v = key_frame(logvar_v, args.sample_rate, device) 
        recon_batch_w, z_w, mu_w, logvar_w = model_w(data_w) 
        recon_batch_w_ne, z_w_ne, mu_w_ne, logvar_w_ne = model_w(data_w_ne)# negative words set， Perform the same encoder operation
        K_v, K_w, K_vw, K_v_ne, K_w_ne, K_vw_ne = k_pae(mu_v, mu_w, mu_w_ne, logvar_v, logvar_w, logvar_w_ne, args, device, g_mat)
    else:
        recon_batch_v, z_v = model_v(data_v) 
        # sample z
        z_v = key_frame(z_v, args.sample_rate, device) 
        recon_batch_w, z_w = model_w(data_w)  
        recon_batch_w_ne, z_w_ne = model_w(data_w_ne)# negative words set， Perform the same encoder operation
        K_v, K_w, K_vw, K_v_ne, K_w_ne, K_vw_ne = k_dae(z_v, z_w, z_w_ne, args, device, g_mat)

    if args.predict_type == 'straight':
        return K_vw.detach().data, 0, 0, K_vw, K_v, K_w

    rec_loss_v = model_v.rec_loss
    # rec_loss_v = loss_function(recon_batch_v, data_v, args.loss_type) # size of recon_batch_v: [I,64], FV feature dimension is 64
    rec_loss_w = loss_function(recon_batch_w, data_w, args.loss_type, transcript, device) # recon_batch_w: [J,48]
    rec_loss = rec_loss_v + rec_loss_w # rec loss

    reg_loss_po, T = ufgw_discrepancy(K_v, K_w, K_vw, device, args, length_distribution_p)# positive reg loss
    if args.enable_contrastive_learning:
        reg_loss_ne, _ = ufgw_discrepancy(K_v_ne, K_w_ne, K_vw_ne, device, args) # negative reg loss
        reg_loss = args.gamma * (reg_loss_po - reg_loss_ne) # reg loss
    else:
        reg_loss = args.gamma * (reg_loss_po) # reg loss

    return T.detach().data, rec_loss.item(), reg_loss.item(), K_vw, K_v, K_w


def test(model_v, model_w, test_data, device, epoch, args, index2label, logdir, savedir, grammar, length_model, mof):
    '''
    test every sample in test_dataset, and compute score 
    '''
    set_gt_labels = []
    set_predict_labels = []
    test_rec_loss = 0
    test_reg_loss = 0
    test_loss = 0
    start = time.time()
    # create dir to save test results
    test_save_dir = os.path.join(logdir, "save_rslt_%d" % epoch)
    os.makedirs(test_save_dir, exist_ok=True)
    # save net params and length model
    network_file_v = savedir + '/network_v.iter-' + str(epoch) + '.net'
    network_file_w = savedir + '/network_w.iter-' + str(epoch) + '.net'
    length_file = savedir + '/lengths.iter-' + str(epoch) + '.txt'
    torch.save(model_v.state_dict(), network_file_v)
    torch.save(model_w.state_dict(), network_file_w)
    np.savetxt(length_file, length_model.mean_lengths)

    # could write this part with multiprocess, to be determined.
    for i, data in enumerate(tqdm(test_data)):
        vfname, sequence, transcript, gt_label = data
        transcript = np.array(transcript)
        # if the task if 'set', transfer the transcript to a complete word set.
        if args.task_type == 'set':
            #transcript = np.arange(len(index2label)) # used in providing nothing in testing type
            transcript = np.array(list(set(transcript)))
        
        # compute grammer distance matrix J*J
        J = transcript.shape[0]
        g_mat = np.zeros((J, J), dtype=np.float32)
        # compute grammar_matrix based on training data transcript
        if args.enable_grammar:
            for i in range(J):
                succ = grammar.successors.get(transcript[i], list())
                succ_num = len(succ)
                for j in range(J):
                    jnum = succ.count(transcript[j])
                    if j == i:
                        g_mat[i][j] = 0
                    elif jnum == 0:
                        g_mat[i][j] = 10*succ_num
                    else:
                        g_mat[i][j] = succ_num / jnum
                
                isum = g_mat[i].sum()
                if isum == 0:
                    continue
                for j in range(J):
                    g_mat[i][j] /= isum

        # compute action lengths distribution as prior
        length_distribution_p = []
        for item in transcript:
            times = np.sum(transcript == item)
            length_distribution_p.append(length_model.mean_lengths[item] / times)
        
        length_distribution_p = np.array(length_distribution_p, dtype=np.float32)
        length_distribution_p /= np.sum(length_distribution_p)
        length_distribution_p = torch.from_numpy(length_distribution_p).to(device)
        length_distribution_p = length_distribution_p.view(len(transcript), 1)

        torch_sequence, torch_word_embedding, torch_word_embedding_complement = transfer_data_to_torch_mlp(sequence, transcript, index2label, args.xw_dim, args.sample_rate)
        
        with torch.no_grad():
            # think differently about transcript and set, the difference here is the word input.
            OT_matrix, rec_loss, reg_loss, K_vw, K_v, K_w = predict(model_v, model_w, torch_sequence, torch_word_embedding, torch_word_embedding_complement, device, epoch, args, g_mat, length_distribution_p, transcript)
        
        # transfer OT_matrix to predict_labels and save generated results.
        predict_labels = decode(vfname, sequence, transcript, gt_label, OT_matrix, index2label, test_save_dir, args)
        set_gt_labels.append(gt_label)
        set_predict_labels.append(predict_labels)
        test_rec_loss += rec_loss
        test_reg_loss += reg_loss
        test_loss += rec_loss + reg_loss
    
    # compute mof, mof-bg, iou, iod
    result = compute_score(set_gt_labels, set_predict_labels)
    mof.append(result['MoF'])
    # output test metrics and loss
    test_data_len = len(test_data)
    test_rec_loss /= test_data_len
    test_reg_loss /= test_data_len
    test_loss /= test_data_len
    duration = (time.time() - start) / 60
    print("-------------------------Test Result. Time %f m" % duration)
    pprint(result)
    print('====> Test set RecLoss: {:.4f} RegLoss: {:.4f} TotalLoss: {:.4f}'.format(
        test_rec_loss, test_reg_loss, test_loss))
    
    # save alignment metrics results by json file 
    with open(os.path.join(test_save_dir, 'alignment_metrics.json'), 'w') as fp:
        json.dump(result, fp)


def train_model(model_v, model_w, train_data, test_data, device, args, label2index, index2label, logdir, savedir, epoch_start=0):
    model_v = model_v.to(device)
    model_w = model_w.to(device)

    optimizer = optim.Adam(list(model_v.parameters()) + list(model_w.parameters()), lr=args.lr, betas=(0.9, 0.999))
    recloss = []
    regloss_po = []
    regloss_ne = []
    totalloss = []
    mof = []
    test_grammar, buffer, length_model = create_tools(train_data, index2label, label2index, savedir, args)

    train_time_all = 0
    test_time_all = 0
    test_times = 0

    for epoch in range(epoch_start + 1, args.M + 1):
        vfname, sequence, transcript, gt_label = train_data.get()
        # remove SIL label in transcript
        try:
            while True:
                transcript.remove(0) # label2index['SIL'] = 0
        except ValueError:
            pass

        transcript = np.array(transcript)
        if args.task_type == 'transcript':
            # if transcript, apply it directly.
            pass
        elif args.task_type == 'set':
            # if set, transfer transcript to set first.
            transcript = np.array(list(set(transcript))) # transcript => action set. when applying to transcript task, should add action relations to distance matrix.
        time1 = time.time()
        train(model_v, model_w, vfname, sequence, transcript, gt_label, index2label, device, optimizer, epoch, args, length_model, buffer, recloss, regloss_po, regloss_ne, totalloss)
        time2 = time.time()
        train_time_all += time2 - time1

        if epoch % args.test_every == 0:
            
            time3 = time.time()
            test(model_v, model_w, test_data, device, epoch, args, index2label, logdir, savedir, test_grammar, length_model, mof)
            test_times += 1
            test_time_all += time.time() - time3


    print('Train time average: {}s.'.format(train_time_all / args.M))
    assert test_times != 0
    print('Test time average: {}s.'.format(test_time_all / (test_times * len(test_data))))
