import torch

rot_bins_loss_coeff = 1
front_loss_coeff = 1
translation_loss_coeff = 4


def roll_by_gather(mat,dim, shifts: torch.LongTensor):

    #print("roll shapes", mat.shape, shifts.shape)

    # assumes 2D array
    n_rows, n_cols = mat.shape
    
    if dim==0:
        #print(mat)
        arange1 = torch.arange(n_rows).view((n_rows, 1)).repeat((1, n_cols)).cuda()
        #print(arange1)
        arange2 = (arange1 - shifts) % n_rows
        #print(arange2)
        return torch.gather(mat, 0, arange2)
    elif dim==1:
        arange1 = torch.arange(n_cols).view(( 1,n_cols)).repeat((n_rows,1)).cuda()
        #print(arange1)
        arange2 = (arange1 - shifts) % n_cols
        #print(arange2)
        return torch.gather(mat, 1, arange2)