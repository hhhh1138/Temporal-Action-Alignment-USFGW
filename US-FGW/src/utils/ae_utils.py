import torch
import torch.nn.functional as F
import numpy as np
import random
from .grammar import PathGrammar
from .buffer import Buffer
from .length_model import PoissonModel
import os


def convert2onehot(X, N):
    '''
    convert word labels to one-hot code

    :param X: np.darray (len, )
    :param N: action nums
    :return X: np.darray : one-hot code (N, len)
    '''
    X = np.eye(N)[X.reshape(-1)].T

    return X.astype(np.float32)


def get_negative_words(transcript, label2index):
    '''
    get the complement of the transcript

    :param transcript: np.darray
    :return neg: np.darray
    '''
    W = len(label2index)
    neg = []
    for i in range(1, W):
        if i not in transcript:
            neg.append(i)

    return np.array(neg)


def key_frame(M, sample_rate=10, device='cuda'):
    '''
    extract key frames every 10 frames randomly. Specially, The last key frame is extracted from last I%10 frames. 
    '''
    r = M.size(0) // sample_rate + (M.size(0) % sample_rate != 0) # number of rows of M (i.e. sequence.T)
    tmp = torch.zeros(r, M.size(1)).to(device)
    for i in range(M.size(0) // sample_rate):
        j = random.randint(0, sample_rate - 1)
        tmp[i , :] = M[sample_rate * i + j , :]
    if M.size(0) % sample_rate != 0:
        tmp[-1 , :] = M[sample_rate * (i + 1) + random.randint(0 , M.size(0) % sample_rate - 1)]
    
    return tmp


def transfer_data_to_torch_mlp(sequence, transcript, index2label, xw_dim, sample_rate):
    '''
    deal with the input data and transfer the tensor type (used in mlp AE)
    '''
    transcript_complement = get_negative_words(transcript, index2label)
    word_embedding = convert2onehot(transcript, xw_dim)
    word_embedding_complement = convert2onehot(transcript_complement, xw_dim)
    # sequence: xv * I => I*xv
    # word_embedding: xw * J => J*xw
    # word_embedding_complement: xw * (all - set(J)) => (all - set(J))*xw
    # For mlp-based encoder and decoder, input data processed like below:
    torch_sequence = torch.from_numpy(sequence.T).float() 
    torch_word_embedding = torch.from_numpy(word_embedding.T)
    torch_word_embedding_complement = torch.from_numpy(word_embedding_complement.T)

    return torch_sequence, torch_word_embedding, torch_word_embedding_complement

def distance_matrix(pts_src: torch.Tensor, pts_dst: torch.Tensor, p: int = 2):
    """
    Returns the matrix of ||x_i-y_j||_p^p.
    :param pts_src: [R, D] matrix
    :param pts_dst: [C, D] matrix
    :param p:
    :return: [R, C] distance matrix
    """
    x_col = pts_src.unsqueeze(1) # [J,1,z_dim] or [I,1,z_dim]
    y_row = pts_dst.unsqueeze(0) # [1,J,z_dim] or [1,I,z_dim]
    distance = torch.sum((torch.abs(x_col - y_row)) ** p, 2)

    return distance


def kernel_pae(mu_src: torch.Tensor, mu_dst: torch.Tensor, logvar_src: torch.Tensor, logvar_dst: torch.Tensor, args, device, type=0, g_mat=None):
    """
    Probabilistic
    Calculate a kernel matrix between the gmm distributions with diagonal variances
    :param mu_src: [R, D] matrix, the means of R Gaussian distributions
    :param mu_dst: [C, D] matrix, the means of C Gaussian distributions
    :param logvar_src: [R, D] matrix, the log(variance) of R Gaussian distributions
    :param logvar_dst: [C, D] matrix, the log(variance) of C Gaussian distributions
    :return: [R, C] distance matrix
    """
    std_src = torch.exp(0.5 * logvar_src) # [I,z_dim] or [J,z_dim]
    std_dst = torch.exp(0.5 * logvar_dst)
    distance_mean = distance_matrix(mu_src, mu_dst, p=2)
    distance_var = distance_matrix(std_src, std_dst, p=2)

    if type == 1:
        Lambda = args.Lambdav
    elif type == 2:
        Lambda = args.Lambdaw
    else:
        Lambda = args.Lambdaw

    A = distance_mean.size(0)
    B = distance_mean.size(1)
    order_info = np.zeros((A, B)).astype(np.float32)
    # if args.task_type == 'transcript' : # consider distance between the normalized indices of frames/texts
    for i in range(A):
        for j in range(B):
            order_info[i][j] = order_info[i][j] + Lambda * (abs(i / A - j / B))
        
    distance_mean = distance_mean + torch.from_numpy(order_info).to(device)
    
    # if g_mat is not None:
    #     g_mat = g_mat * 0.1
    #     distance_mean = distance_mean + torch.from_numpy(g_mat).to(device) 

    kernel = torch.exp(-1 / args.b * (distance_mean + distance_var + 1e-6)) if args.enable_spectral else distance_mean + distance_var

    return  kernel

def kernel_dae(z_src: torch.Tensor, z_dst: torch.Tensor, args, device, type=0, g_mat=None):
    """
    Deterministic
    Calculate a kernel matrix between the gmm distributions with diagonal variances
    :param z_src: [R, D] matrix, the latent codes of R Gaussian distributions
    :param z_dst: [C, D] matrix, the latent codes of C Gaussian distributions
    :return: [R, C] distance matrix
    """
    distance_latent = distance_matrix(z_src, z_dst, p=2) # paper 520-522 

    if type == 1:
        Lambda = args.Lambdav
    elif type == 2:
        Lambda = args.Lambdaw
    else:
        Lambda = args.Lambdaw

    A = distance_latent.size(0)
    B = distance_latent.size(1)
    order_info = np.zeros((A, B)).astype(np.float32)
    #if args.task_type == 'transcript' : # consider distance between the normalized indices of frames/texts
    for i in range(A):
        for j in range(B):
            order_info[i][j] = order_info[i][j] + Lambda * (abs(i / A - j / B))
        
    distance_latent = distance_latent + torch.from_numpy(order_info).to(device)

    # if g_mat is not None:
    #     g_mat = g_mat * 0.1
    #     distance_latent = distance_latent + torch.from_numpy(g_mat).to(device) 

    kernel = torch.exp(-1 / args.b * (distance_latent + 1e-6)) if args.enable_spectral else distance_latent

    return  kernel


def k_pae(mu_v, mu_w, mu_w_ne, logvar_v, logvar_w, logvar_w_ne, args, device, g_mat=None):
    K_v = kernel_pae(mu_v, mu_v, logvar_v, logvar_v, args, device, 1) # [I,I]
    K_w = kernel_pae(mu_w, mu_w, logvar_w, logvar_w, args, device, 2, g_mat) # [J,J]
    K_vw = kernel_pae(mu_v, mu_w, logvar_v, logvar_w, args, device) # [I,J]
    # K_v_ne = kernel_pae(mu_v, mu_v, logvar_v, logvar_v, args) # [I,I]
    K_w_ne = kernel_pae(mu_w_ne, mu_w_ne, logvar_w_ne, logvar_w_ne, args, device, 2) # [J,J]
    K_vw_ne = kernel_pae(mu_v, mu_w_ne, logvar_v, logvar_w_ne, args, device) # paper 520-522 [I,J]

    return K_v, K_w, K_vw, K_v, K_w_ne, K_vw_ne

def k_dae(z_v, z_w, z_w_ne, args, device, g_mat=None):
    K_v = kernel_dae(z_v, z_v, args, device, 1) # [I,I]
    K_w = kernel_dae(z_w, z_w, args, device, 2, g_mat) # [J,J]
    K_vw = kernel_dae(z_v, z_w, args, device) # [I,J]
    # K_v_ne = kernel_dae(z_v, z_v, args) # [I,I]
    K_w_ne = kernel_dae(z_w_ne, z_w_ne, args, device, 2) # [J,J]
    K_vw_ne = kernel_dae(z_v, z_w_ne, args, device) # paper 520-522 [I,J]

    return K_v, K_w, K_vw, K_v, K_w_ne, K_vw_ne

def bregman_admm_iteration(K_vw: torch.Tensor, K_v: torch.Tensor, K_w: torch.Tensor, 
                            p_v: torch.Tensor, p_w: torch.Tensor, trans0: torch.Tensor,
                            beta: float = 0.1, tau: float = 0.1, rho: float = 1.0, 
                            error_bound: float = 1e-3, max_iter: int = 50, device: str = 'cuda', 
                            length_distribution: torch.Tensor = None) -> torch.Tensor:
    '''
    Bregman-ADMM iteration algorithm

    Args:
        K_vw: (I, J) array representing distance between nodes
        K_v: (I, I) array representing distance between nodes
        K_w: (J, J) array representing distance between nodes
        p_v: (I, 1) array representing the distribution of source nodes
        p_w: (J, 1) array representing the distribution of target nodes
        trans0: (I, J) initial array of optimal transport
        beta: the trade-off between w and gw in fgw
        tau: the weight of entropic regularizer
        rho: a hyperpara controlling rate of convergence in badmm
        error_bound: the error bound to check convergence
        max_iter: the maximum number of iterations
        device: running on cuda or cpu
    returns:
        T: final OT matrix
    
    '''
    I = K_vw.size(0)
    J = K_vw.size(1)

    if p_v is None:
        p_v = torch.ones(I, 1) / I
        p_v = p_v.to(device)
    
    if p_w is None:
        p_w = torch.ones(J, 1) / J
        p_w = p_w.to(device)

    if trans0 == None:
        trans0 = p_v @ torch.t(p_w)
        trans0 = trans0.to(device)
    
    trans1 = torch.zeros(trans0.shape).to(device)
    u = torch.ones(I,1).to(device) / I
    mu = torch.ones(J,1).to(device) / J
    Z = torch.zeros(trans0.shape).to(device)
    z1 = torch.zeros(u.shape).to(device)
    z2 = torch.zeros(mu.shape).to(device)

    relative_error = error_bound + 1.0
    i = 1

    while relative_error > error_bound and i <= max_iter:
        
        tmp = ((1 - beta) * K_vw + beta * (K_v @ trans0 @ torch.t(K_w)) + rho * torch.log(trans0) - Z) / rho
        trans1 = torch.diag(u.reshape(-1)) @ F.softmax(tmp, dim=1)   # T^{k+1}

        tmp = (beta * (torch.t(K_v) @ trans1 @ K_w) + Z + rho * torch.log(trans1)) / rho
        trans0 = F.softmax(tmp, dim=0) @ torch.diag(mu.reshape(-1))   # S^{k+1}

        tmp = (rho * torch.log(torch.sum(trans1, dim=1, keepdim=True)) + tau * torch.log(torch.ones(I, 1).to(device) / I) - z1) / (rho + tau)
        u = F.softmax(tmp, dim=0)                              # u^{k+1}

        if length_distribution is None:
            tmp = (rho * torch.log(torch.sum(torch.t(trans0), dim=1, keepdim=True)) + tau * torch.log(torch.ones(J, 1).to(device) / J) - z2) / (rho + tau)
        else:
            tmp = (rho * torch.log(torch.sum(torch.t(trans0), dim=1, keepdim=True)) + tau * torch.log(length_distribution) - z2) / (rho + tau)
        mu = F.softmax(tmp, dim=0)                             # mu^{k+1}

        Z = Z + rho * (trans1 - trans0)
        z1 = z1 + rho * (u - torch.sum(trans1, dim=1, keepdim=True))
        z2 = z2 + rho * (mu - torch.sum(torch.t(trans0), dim=1, keepdim=True))
        
        relative_error = torch.sum(torch.abs(trans1.detach().data - trans0.detach().data)) / torch.sum(torch.abs(trans1.detach().data)) + \
                            torch.sum(torch.abs(u.detach().data - torch.sum(trans1.detach().data, dim=1, keepdim=True))) / torch.sum(torch.abs(u.detach().data)) + \
                               torch.sum(torch.abs(mu.detach().data - torch.sum(torch.t(trans0.detach().data), dim=1, keepdim=True))) / torch.sum(torch.abs(mu.detach().data))
        
        i += 1
    
    return trans1


def ufgw_discrepancy(K_v, K_w, K_vw, device, args, length_distribution=None):
    '''
    compute usfgw distance and OT matrix
    '''
    # initial OT matrix
    I = K_v.size(0) 
    J = K_w.size(0) 
    T = torch.ones(I, J) / (I * J) # [I,J]
    T = T.to(device)
    # compute OT matrix by B-ADMM
    T = bregman_admm_iteration(K_vw, K_v, K_w, None, None, T, args.beta, args.tau, args.badmm_rho, args.badmm_error_bound, args.badmm_loops, device, length_distribution)
    if torch.isnan(T).sum() > 0:
        T = (torch.ones(I, J) / (I * J)).to(device)
    
    cost_1 = -K_vw if args.enable_spectral else K_vw
    cost_2 = -K_v @ T.detach().data @ torch.t(K_w)
    d_ufgw = (1 - args.beta) * (cost_1 * T.detach().data).sum() + args.beta * (cost_2 * T.detach().data).sum()

    return d_ufgw, T.detach().data


def create_tools(train_data, index2label, label2index, savedir, args):
    '''
    create tools : grammar, buffer, length_model 
    '''
    paths = set()
    for _, _, transcript, gt_label in train_data:
        paths.add( ' '.join([index2label[index] for index in transcript]) )
    with open(os.path.join(savedir, 'grammar.txt'), 'w') as f:
        f.write('\n'.join(paths) + '\n')
    
    test_grammar = PathGrammar(os.path.join(savedir, 'grammar.txt'), label2index)

    buffer = Buffer(buffer_size=len(train_data), n_classes=train_data.n_classes)
    
    maxlen = 80 if args.data == "crosstask" else 2000 # by default, limit all actions to 2000 or 80
    bg_limit = 160 
    # replace the default_model with a path str if u want to load your saved length model.
    defaut_model = np.ones(len(label2index), dtype=np.float32) 
    length_model = PoissonModel(defaut_model, max_length=maxlen, bg_limit=bg_limit)
    
    return test_grammar, buffer, length_model
