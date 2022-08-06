import time
import torch
import numpy as np

# def calculate_contrast(img1,img2,window_size):
#     #img1 B, 3,256,256
#     print(img1.shape)
#     img1 = img1.permute(0, 2, 3, 1)
#     img2 = img2.permute(0, 2, 3, 1)
#     batch = img1.shape[0]
#     #print(img1.shape,img2.shape)
#     d_1 = calculate_contrast_oneimg(img1,window_size)
#     d_2 = calculate_contrast_oneimg(img2,window_size)
#     diff_img = abs(d_1)-abs(d_2)
#     diff_img = diff_img * diff_img
#     result = torch.sum(torch.sum(torch.sum(torch.sum(diff_img,3),2),1),0)/(((window_size*2+1)*(window_size*2+1)-1)*((window_size*2+1)*(window_size*2+1)-1))
#     return result

def calculate_contrast_oneimg(img,window_size):
    img = img.permute(0, 2, 3, 1)
    x_diff = img[:, window_size:256 - window_size, window_size:256 - window_size]
    x = x_diff
    #print('2222',x.shape)
    start = time.time()
    flag = 0
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            if (i == -window_size) and (j == -window_size):
                img_diff = x_diff - img[:, window_size + i:256 - window_size + i, window_size + j:256 - window_size + j]
                img_diff = torch.sum(img_diff * img_diff, 3)
                img_diff = torch.unsqueeze(img_diff, 3)
                x = img_diff
                flag += 1
            elif (i == 0) and (j == 0):
                continue
            else:
                # print(img_diff.shape)
                flag += 1
                # print(i, j, flag)
                img_diff = x_diff - img[:, window_size + i:256 - window_size + i, window_size + j:256 - window_size + j]
                img_diff = torch.sum(img_diff * img_diff, 3)
                img_diff = torch.unsqueeze(img_diff, 3)
                x = torch.cat((x, img_diff), 3)

    nrand = np.array([i for i in range(120)])
    #np.random.shuffle(nrand)
    #print(np.random.shuffle(nrand))
    #nrand = nrand[0:60]
    #print(nrand)
    trand = torch.from_numpy(nrand).type(torch.long)
    #trand = torch.randint(0, 120, (60,))
    return abs(x[:, :, :, trand])

def calculate_contrast_oneimg_l1(img,window_size):
    img = img.permute(0, 2, 3, 1)
    x_diff = img[:, window_size:256 - window_size, window_size:256 - window_size]
    x = x_diff
    #print('2222',x.shape)
    start = time.time()
    flag = 0
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            if (i == -window_size) and (j == -window_size):
                img_diff = x_diff - img[:, window_size + i:256 - window_size + i, window_size + j:256 - window_size + j]
                img_diff = torch.sum(torch.abs(img_diff), 3)
                #print(img_diff,img_diff.size())
                img_diff = torch.unsqueeze(img_diff, 3)
                x = img_diff
                flag += 1
            elif (i == 0) and (j == 0):
                continue
            else:
                # print(img_diff.shape)
                flag += 1
                # print(i, j, flag)
                img_diff = x_diff - img[:, window_size + i:256 - window_size + i, window_size + j:256 - window_size + j]
                img_diff = torch.sum(torch.abs(img_diff), 3)
                img_diff = torch.unsqueeze(img_diff, 3)
                x = torch.cat((x, img_diff), 3)
    # print(x[1,120,120,:])
    # exit()
    nrand = np.array([i for i in range(120)])
    #np.random.shuffle(nrand)
    #print(np.random.shuffle(nrand))
    #nrand = nrand[0:60]
    #print(nrand)
    trand = torch.from_numpy(nrand).type(torch.long)
    #trand = torch.randint(0, 120, (60,))
    return abs(x[:, :, :, trand])
def global_contrast_img(img,img2,points_number=5):
    img = img.permute(0, 2, 3, 1)
    img2 = img2.permute(0,2,3,1)

    hight, width = img.shape[1],img.shape[2]

    select_points = torch.tensor(np.zeros((img.size(0), *(1,points_number,1))))

    rand_hight = torch.randint(0, hight, (points_number,))
    rand_width = torch.randint(0, width, (points_number,))

    rand_hight1 = torch.randint(0, hight, (points_number,))
    rand_width1= torch.randint(0, width, (points_number,))

    img_points1 = img[:,rand_width,rand_hight,:]
    #print(img_points1.shape)
    img_points2 = img[:, rand_width1, rand_hight1, :]
    img1_diff = (img_points1 - img_points2) *(img_points1 - img_points2)
    img1_diff = torch.sum(img1_diff, 2)

    img2_points1 = img2[:,rand_width,rand_hight,:]
    img2_points2 = img2[:, rand_width1, rand_hight1, :]

    img2_diff = (img2_points1 - img2_points2) * (img2_points1 - img2_points2)
    img2_diff = torch.sum(img2_diff, 2)
    #print(img1_diff.shape)

    #print(img1_diff)
    #
    # for i in range(points_number):
    #     rand_hight = torch.randint(0, hight,(points_number,))
    #     rand_width = torch.randint(0, width,(points_number,))
    #
    #     #print(rand_hight,rand_width,select_points.size())
    #     select_points[:,:,i,:] = img[:,rand_hight,rand_width,:]
    # # rand_hight = torch.randint(0, hight,(points_number,))
    # # rand_width = torch.randint(0, width,(points_number,))
    # # print(rand_hight,rand_width,select_points.size())
    # # select_points[:,:,:,:] = img[:,rand_hight,rand_width,:]
    #
    # select_points.permute(0,3,1,2)

    return img1_diff,img2_diff

def global_contrast_img_l1(img,img2,points_number=5):
    img = img.permute(0, 2, 3, 1)
    img2 = img2.permute(0,2,3,1)

    hight, width = img.shape[1],img.shape[2]

    select_points = torch.tensor(np.zeros((img.size(0), *(1,points_number,1))))

    rand_hight = torch.randint(0, hight, (points_number,))
    rand_width = torch.randint(0, width, (points_number,))

    rand_hight1 = torch.randint(0, hight, (points_number,))
    rand_width1= torch.randint(0, width, (points_number,))

    img_points1 = img[:,rand_width,rand_hight,:]
    #print(img_points1.shape)
    img_points2 = img[:, rand_width1, rand_hight1, :]
    img1_diff = (img_points1 - img_points2) #*(img_points1 - img_points2)
    img1_diff = torch.sum(torch.abs(img1_diff), 2)

    img2_points1 = img2[:,rand_width,rand_hight,:]
    img2_points2 = img2[:, rand_width1, rand_hight1, :]

    img2_diff = (img2_points1 - img2_points2) #* (img2_points1 - img2_points2)
    img2_diff = torch.sum(torch.abs(img2_diff), 2)
    #print(img1_diff.shape)

    #print(img1_diff)
    #
    # for i in range(points_number):
    #     rand_hight = torch.randint(0, hight,(points_number,))
    #     rand_width = torch.randint(0, width,(points_number,))
    #
    #     #print(rand_hight,rand_width,select_points.size())
    #     select_points[:,:,i,:] = img[:,rand_hight,rand_width,:]
    # # rand_hight = torch.randint(0, hight,(points_number,))
    # # rand_width = torch.randint(0, width,(points_number,))
    # # print(rand_hight,rand_width,select_points.size())
    # # select_points[:,:,:,:] = img[:,rand_hight,rand_width,:]
    #
    # select_points.permute(0,3,1,2)

    return img1_diff,img2_diff

def global_contrast_img_list(img,img2_list,points_number=5):


    img = img.permute(0, 2, 3, 1)
    for i in range(len(img2_list)):
        img2_list[i] =img2_list[i].permute(0,2,3,1)

    hight, width = img.shape[1],img.shape[2]

    select_points = torch.tensor(np.zeros((img.size(0), *(1,points_number,1))))

    rand_hight = torch.randint(0, hight, (points_number,))
    rand_width = torch.randint(0, width, (points_number,))

    rand_hight1 = torch.randint(0, hight, (points_number,))
    rand_width1= torch.randint(0, width, (points_number,))

    img_points1 = img[:,rand_width,rand_hight,:]
    #print(img_points1.shape)
    img_points2 = img[:, rand_width1, rand_hight1, :]
    img1_diff = (img_points1 - img_points2) *(img_points1 - img_points2)
    img1_diff = torch.sum(img1_diff, 2)

    img2_diff = []
    for i in range(len(img2_list)):
        img2_points1 = img2_list[i][:,rand_width,rand_hight,:]
        img2_points2 = img2_list[i][:, rand_width1, rand_hight1, :]
        temp_img2_diff = (img2_points1 - img2_points2) * (img2_points1 - img2_points2)
        temp_img2_diff = torch.sum(temp_img2_diff, 2)
        img2_diff.append(temp_img2_diff)

    for i in range(len(img2_list)):
        img2_list[i] = img2_list[i].permute(0, 3, 1, 2)
        #print(img1_diff.shape)

    #print(img1_diff)
    #
    # for i in range(points_number):
    #     rand_hight = torch.randint(0, hight,(points_number,))
    #     rand_width = torch.randint(0, width,(points_number,))
    #
    #     #print(rand_hight,rand_width,select_points.size())
    #     select_points[:,:,i,:] = img[:,rand_hight,rand_width,:]
    # # rand_hight = torch.randint(0, hight,(points_number,))
    # # rand_width = torch.randint(0, width,(points_number,))
    # # print(rand_hight,rand_width,select_points.size())
    # # select_points[:,:,:,:] = img[:,rand_hight,rand_width,:]
    #
    # select_points.permute(0,3,1,2)

    return img1_diff,img2_diff

def global_contrast_img_list_l1(img,img2_list,points_number=5):


    img = img.permute(0, 2, 3, 1)
    for i in range(len(img2_list)):
        img2_list[i] =img2_list[i].permute(0,2,3,1)

    hight, width = img.shape[1],img.shape[2]

    select_points = torch.tensor(np.zeros((img.size(0), *(1,points_number,1))))

    rand_hight = torch.randint(0, hight, (points_number,))
    rand_width = torch.randint(0, width, (points_number,))

    rand_hight1 = torch.randint(0, hight, (points_number,))
    rand_width1= torch.randint(0, width, (points_number,))

    img_points1 = img[:,rand_width,rand_hight,:]
    #print(img_points1.shape)
    img_points2 = img[:, rand_width1, rand_hight1, :]
    img1_diff = (img_points1 - img_points2)
    img1_diff = torch.sum(torch.abs(img1_diff), 2)

    img2_diff = []
    for i in range(len(img2_list)):
        img2_points1 = img2_list[i][:,rand_width,rand_hight,:]
        img2_points2 = img2_list[i][:, rand_width1, rand_hight1, :]
        temp_img2_diff = (img2_points1 - img2_points2)
        temp_img2_diff = torch.sum(torch.abs(temp_img2_diff), 2)
        img2_diff.append(temp_img2_diff)

    for i in range(len(img2_list)):
        img2_list[i] = img2_list[i].permute(0, 3, 1, 2)
        #print(img1_diff.shape)

    #print(img1_diff)
    #
    # for i in range(points_number):
    #     rand_hight = torch.randint(0, hight,(points_number,))
    #     rand_width = torch.randint(0, width,(points_number,))
    #
    #     #print(rand_hight,rand_width,select_points.size())
    #     select_points[:,:,i,:] = img[:,rand_hight,rand_width,:]
    # # rand_hight = torch.randint(0, hight,(points_number,))
    # # rand_width = torch.randint(0, width,(points_number,))
    # # print(rand_hight,rand_width,select_points.size())
    # # select_points[:,:,:,:] = img[:,rand_hight,rand_width,:]
    #
    # select_points.permute(0,3,1,2)

    return img1_diff,img2_diff
# img = torch.rand((10,256,256,3)).cuda()
# window_size = 5
#
# x_diff = img[:, window_size:256 - window_size, window_size:256 - window_size]
# x = x_diff
# # print(x.shape)
# start = time.time()
# flag = 0
# for i in range(-window_size, window_size + 1):
#     for j in range(-window_size, window_size + 1):
#         if (i == -window_size) and (j == -window_size):
#
#             img_diff = x_diff - img[:, window_size + i:256 - window_size + i, window_size + j:256 - window_size + j]
#             img_diff = torch.sum(img_diff * img_diff, 3)
#             img_diff = torch.unsqueeze(img_diff, 3)
#             x = img_diff
#             flag += 1
#         elif (i == 0) and (j == 0):
#             continue
#         else:
#             # print(img_diff.shape)
#             flag += 1
#             #print(i, j, flag)
#             x = torch.cat((x, img_diff), 3)
# trand = torch.randint(0,120,(20,))
#
# print(x.shape)
# print(x[:,:,:,trand],x[:,:,:,trand].shape)
# end = time.time()
#
# print(end-start)
#
# img1 = torch.rand((10,256,256,3)).cuda()
# window_size = 5
# img2 = torch.rand((10,256,256,3)).cuda()
# time1 = time.time()
# result = calculate_contrast(img1,img2,window_size)
# time2 = time.time()
# print(time2-time1,result)