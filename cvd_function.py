import numpy as np
import torch
import copy
def cvd_matrix(cvd_type, cvd_level):

    if cvd_type == 0 :
        T={
            10: [ 0.856167,  0.182038, -0.038205,
                  0.029342,  0.955115,  0.015544,
                  -0.002880
                 , - 0.001563,  1.004443],

            20: [ 0.734766,  0.334872, -0.069637,
                  0.051840,  0.919198,  0.028963,
                  -0.004928, -0.004209,  1.009137],

            30: [0.630323, 0.465641, -0.095964,
                 0.069181, 0.890046, 0.040773,
                 -0.006308, -0.007724, 1.014032],
            40:  [0.539009, 0.579343, -0.118352,
                  0.082546, 0.866121, 0.051332,
                  -0.007136, -0.011959, 1.019095],
            50:[0.458064, 0.679578, -0.137642,
                0.092785, 0.846313, 0.060902,
                -0.007494, -0.016807, 1.024301],
            60: [0.385450, 0.769005, -0.154455,
                 0.100526, 0.829802, 0.069673,
                 -0.007442, -0.022190, 1.029632],
            70:[0.319627, 0.849633, -0.169261,
                0.106241, 0.815969, 0.077790,
                -0.007025, -0.028051, 1.035076],
            80:[0.259411, 0.923008, -0.182420,
                0.110296, 0.804340, 0.085364,
                -0.006276, -0.034346, 1.040622],
            90:[0.203876, 0.990338, -0.194214,
                0.112975, 0.794542, 0.092483,
                -0.005222, -0.041043, 1.046265],
            100:[0.152286, 1.052583, -0.204868,
                 0.114503, 0.786281, 0.099216,
                 -0.003882, -0.048116, 1.051998]
        }
        return T.get(cvd_level,None)
    elif cvd_type == 1:
        T={
            10: [0.866435, 0.177704, -0.044139,
                 0.049567, 0.939063, 0.011370,
                 -0.003453, 0.007233, 0.996220],
            20: [0.760729, 0.319078, -0.079807,
                 0.090568, 0.889315, 0.020117,
                 -0.006027, 0.013325, 0.992702],
            30: [0.675425, 0.433850, -0.109275,
                 0.125303, 0.847755, 0.026942,
                 -0.007950, 0.018572, 0.989378],
            40: [0.605511, 0.528560, -0.134071,
                 0.155318, 0.812366, 0.032316,
                 -0.009376, 0.023176, 0.986200],
            50: [0.547494, 0.607765, -0.155259,
                 0.181692, 0.781742, 0.036566,
                 -0.010410, 0.027275, 0.983136],
            60: [0.498864, 0.674741, -0.173604,
                 0.205199, 0.754872, 0.039929,
                 -0.011131, 0.030969, 0.980162],
            70: [0.457771, 0.731899, -0.189670,
                 0.226409, 0.731012, 0.042579,
                 -0.011595, 0.034333, 0.977261],
            80: [0.422823, 0.781057, -0.203881,
                 0.245752, 0.709602, 0.044646,
                 -0.011843, 0.037423, 0.974421],
            90: [0.392952, 0.823610, -0.216562,
                 0.263559, 0.690210, 0.046232,
                 -0.011910, 0.040281, 0.971630],
            100: [0.367322, 0.860646, -0.227968,
                  0.280085, 0.672501, 0.047413,
                  -0.011820, 0.042940, 0.968881],
        }
        return T.get(cvd_level,None)
    elif cvd_type == 2:
        T={
            20: [0.895720, 0.133330, -0.029050,
                 0.029997, 0.945400, 0.024603,
                 0.013027, 0.104707, 0.882266],
            40: [0.948035, 0.089490, -0.037526,
                 0.014364, 0.946792, 0.038844,
                 0.010853, 0.193991, 0.795156],
            60: [1.104996, -0.046633, -0.058363,
                 -0.032137, 0.971635, 0.060503,
                 0.001336, 0.317922, 0.680742],
            80: [1.257728, -0.139648, -0.118081,
                 -0.078003, 0.975409, 0.102594,
                 -0.003316, 0.501214, 0.502102],
            100: [1.255528, -0.076749, -0.178779,
                  -0.078411, 0.930809, 0.147602,
                  0.004733, 0.691367, 0.303900],
        }
        return T.get (cvd_level,None)


def cvd_simulate(I, T, choose=0):
    if choose == 0:
        [img_h, img_w, img_depth] = I.shape
        # I = I.astype(float)
        O = np.dot(np.reshape(I, (img_h * img_w, 3)), T.transpose())
        O = np.reshape(O, (img_h, img_w, 3))
        O = range_truncate(O,0,1)
    if choose == 1:
        B = I[:, :, 0]
        G = I[:, :, 1]
        R = I[:, :, 2]
        img_rgb = copy.deepcopy(I)
        img_rgb[:, :, 0] = R  # [R,G,B]
        img_rgb[:, :, 1] = G
        img_rgb[:, :, 2] = B
        [img_h, img_w, img_depth] = I.shape
        I = img_rgb.astype(float)
        O = np.dot(np.reshape(I, (img_h * img_w, 3)), T.transpose())
        O = np.reshape(O, (img_h, img_w, 3))
        O = range_truncate(O,0,1)

    return O


def range_truncate(I, lowlmt, uplmt):
#     % Description
#     % - Truncate those elements that are in matrix I
#     %   which are less than lower limit to lower limit
#     %   and greater than upper limit to upper limit.

#     % Input parameters
#     % I      = double matrix
#     % lowlmt = lower limit
#     % uplmt  = upper limit

    N = I.shape[0]
    if(N < 1):
        return
    if lowlmt > uplmt:
        return
    O = copy.deepcopy(I)
    O[I < lowlmt] = lowlmt
    O[I > uplmt]  = uplmt

    return O

# pt = torch.rand(10, 3, 256, 256)
# cvd_type = 0
# degree = 100
# T = cvd_matrix(cvd_type,degree)
# T = np.reshape(T,(3,3))
# t2 = torch.tensor(T)
# t3 = t2.unsqueeze(0)
# t3 = t3.repeat([10,1,1])
# #print(t2.size(),t3.size())
# #t1 = torch.from_numpy(T)
# #torch.tensor(10*t1)
# #t1= t1 *(10,1,1)
# print(type(t1),t1.size())
# T.tensor()
# img = pt[0]
# #img = img.review(3,256*256)
# img = img.permute(1,2,0)
# img_np = img.numpy()
# cvd_img = cvd_simulate(img_np, T)
# cvd_img = range_truncate(cvd_img,0,1)


def cvd_simulation_tensors(img, cvd_type, degree):
    T = cvd_matrix(cvd_type, degree)
    T = np.reshape(T, (3, 3))
    T = T.transpose()
    T_tensor = torch.tensor(T)
    T_tensor = T_tensor.unsqueeze(0)
    T_tensor = T_tensor.repeat([img.shape[0], 1, 1])
    T_tensor = T_tensor.to()
        # cuda()
    #imgs["B"].type(Tensor))
    T_tensor = T_tensor.type(torch.cuda.FloatTensor)

    h, w = img.shape[2],img.shape[3]
    img = img.view([-1 ,3 ,h*w])

    img = img.permute(0, 2, 1) # B H*W C
    #img = img.([[img.shape[0], -1, , 256])
    #print(111,img.size(),T_tensor.size())
    #print(img.type(),T_tensor.type())
    cvd_img = torch.bmm(img,T_tensor)
    cvd_img = cvd_img.permute(0, 2, 1) # B  C  H*W
    cvd_img = cvd_img.view([-1,3,h,w])


    out_put = cvd_img.clone()
    out_put[cvd_img < 0] = 0
    out_put[cvd_img > 1] = 1
    return out_put
    # for i in range(img.shape[0]):
    #     # img = img.review(3,256*256)
    #     img1 = img[i]# c h w
    #     img1 = img1.permute(1, 2, 0) #  h w c
    #     img_np = img1.numpy()
    #     cvd_img = cvd_simulate(img_np, T)
    #     cvd_img = range_truncate(cvd_img, 0, 1)
    #     img[i] = cvd_img.permute(2, 0, 1) # c h w
    #
    # [img_h, img_w, img_depth] = I.shape
    # # I = I.astype(float)
    # O = np.dot(np.reshape(I, (img_h * img_w, 3)), T.transpose())
    # O = np.reshape(O, (img_h, img_w, 3))

    return img