import numpy as np
import math
import sys


def gradient_filter(img):

    # Finding both vertical and horizontal gradients using sobel operator
    # Summing them up to achieve energy map (gradient magnitude)

    gradient = np.zeros(img.shape,dtype = np.int32)
    max_x, max_y = img.shape

    for y in range(1, max_y - 1):
        for x in range(1, max_x - 1):
            dx_pos = 4 * img[x, y] - 2 * img[x - 1, y] - img[x - 1, y + 1] - img[x - 1, y - 1]
            dx_neg = 4 * img[x, y] - 2 * img[x + 1, y] - img[x + 1, y + 1] - img[x + 1, y - 1]
            dx = dx_pos - dx_neg

            dy_pos = 4 * img[x, y] - 2 * img[x, y - 1] - img[x + 1, y - 1] - img[x - 1, y - 1]
            dy_neg = 4 * img[x, y] - 2 * img[x, y + 1] - img[x + 1, y + 1] - img[x - 1, y + 1]
            dy = dy_pos - dy_neg

            gradient[x, y] = math.fabs(dx) + math.fabs(dy)

    return gradient


def find_vertical_seam(energy_img):

    inv_seam, min_seam_value = find_horizontal_seam(energy_img.T)
    seam_pixels = []
    for rev_pix in inv_seam:
        seam_pixels.append((rev_pix[1],rev_pix[0]))

    # print min_seam_value
    return seam_pixels,min_seam_value


def find_horizontal_seam(energy_img):
    optimal_cost_dp, path_table = get_DPtable_PathTable(energy_img)

    final_column = list(optimal_cost_dp[:,energy_img.shape[1]-1])
    # print optimal_cost_dp
    # print path_table
    min_seam_value, min_seam_index = get_min_minIndex(final_column)
    # print final_column
    # print min_seam_index,min_seam_value
    seam_pixels = construct_path_horizontal(path_table,min_seam_index)
    # print seam_pixels
    return seam_pixels, min_seam_value

def get_DPtable_PathTable(energy_img):
    optimal_cost_dp = np.zeros(energy_img.shape)
    path_table = np.zeros(energy_img.shape)

    for i in range(energy_img.shape[1]):
        for j in range(energy_img.shape[0]):
            if i == 0:
                optimal_cost_dp[j][i] = energy_img[j][i]
                continue

            if j == 0:
                min_value, min_index = get_min_minIndex((optimal_cost_dp[j][i - 1], optimal_cost_dp[j + 1][i - 1]))
                optimal_cost_dp[j][i] = energy_img[j][i] + min_value
                path_table[j][i] = 1 + min_index
                continue
            if j == energy_img.shape[0] - 1:
                min_value, min_index = get_min_minIndex((optimal_cost_dp[j - 1][i - 1], optimal_cost_dp[j][i - 1]))
                optimal_cost_dp[j][i] = energy_img[j][i] + min_value
                path_table[j][i] = min_index
                continue

            min_value, min_index = \
                get_min_minIndex(
                    (optimal_cost_dp[j - 1][i - 1], optimal_cost_dp[j][i - 1], optimal_cost_dp[j + 1][i - 1]))

            optimal_cost_dp[j][i] = energy_img[j][i] + min_value
            path_table[j][i] = min_index

    return optimal_cost_dp,path_table



def get_min_minIndex(lis):

    length = len(lis)
    if length == 3:
        if lis[0]==lis[1] and lis[0]<lis[2]:
            return lis[1],1
        if lis[1]==lis[2] and lis[1]<lis[0]:
            return lis[1],1
        if lis[0] == lis[1] and lis[0] == lis[2]:
            return lis[1], 1

    min_val = sys.maxint
    min_index = 0
    for i in range(length):
        val = lis[i]
        if val <= min_val:
            min_val = val
            min_index = i

    return min_val,min_index

def construct_path_horizontal(path_table, min_index):

    height, length = path_table.shape
    list_of_pixels = []
    i = length - 1
    j = min_index

    while (i != 0):
        list_of_pixels.append((j,i))
        if path_table[j][i] == 0:
            j = j-1
        elif path_table[j][i] == 1:
            j = j
        elif path_table[j][i] == 2:
            j = j+1

        i = i - 1

    list_of_pixels.append((j,0))

    list_of_pixels.reverse()
    return list_of_pixels


def delete_seam(image,seam_pixels,isHorizontal):
    if len(image.shape) == 3:
        height, length, _ = image.shape
        if isHorizontal:
            out_image = np.zeros((image.shape[0] - 1, image.shape[1], 3), dtype=np.uint8)
        else:
            out_image = np.zeros((image.shape[0], image.shape[1]-1, 3), dtype=np.uint8)

    else:
        height, length = image.shape
        if isHorizontal:
            out_image = np.zeros((image.shape[0] - 1, image.shape[1]), dtype=np.int32)
        else:
            out_image = np.zeros((image.shape[0], image.shape[1]-1), dtype=np.int32)

    if isHorizontal:
        for i in range(length):
            bad_pixel_index = seam_pixels[i][0]
            for j in range(height):
                if j < bad_pixel_index:
                    out_image[j][i] = image[j][i]
                if j > bad_pixel_index:
                    out_image[j - 1][i] = image[j][i]

    else:
        for i in range(height):
            bad_pixel_index = seam_pixels[i][1]
            for j in range(length):
                if j < bad_pixel_index:
                    out_image[i][j] = image[i][j]
                if j > bad_pixel_index:
                    out_image[i][j-1] = image[i][j]

    return out_image

def get_optimal_order_image(energy_img,final_img,req_len,req_height):
    img_height, img_length = energy_img.shape

    r = img_height - req_height
    c = img_length - req_len

    e_img = np.copy(energy_img)
    f_image = np.copy(final_img)

    t_dp = [[0 for x in range(img_length+1)] for y in range(img_height+1)]

    # path_table = np.zeros((r+1,c+1))
    # path_table[0][0] = -1
    t_dp[0][0] = (0,e_img,f_image)

    for i in range(1,r+1):
        seam, min_seam_value = find_horizontal_seam(e_img)
        e_img = delete_seam(e_img,seam,isHorizontal=True)
        f_image = delete_seam(f_image,seam,isHorizontal=True)

        t_dp[i][0] = (t_dp[i-1][0][0] + min_seam_value,e_img,f_image)
        # path_table[i][0] = 1

    e_img = np.copy(energy_img)
    f_image = np.copy(final_img)

    for i in range(1,c+1):
        seam, min_seam_value = find_vertical_seam(e_img)
        e_img = delete_seam(e_img,seam,isHorizontal=False)
        f_image = delete_seam(f_image,seam,isHorizontal=False)
        t_dp[0][i] = (t_dp[0][i-1][0] + min_seam_value,e_img,f_image)
        # path_table[0][i] = 0
    count = 0
    for i in range(1,r+1):
        for j in range(1,c+1):
            count = count + 1
            print (count),
            seamHor, min_seam_valueHor = find_horizontal_seam(t_dp[i-1][j][1])
            hor_value = min_seam_valueHor + t_dp[i-1][j][0]
            count = count + 1
            print (count),
            seamVer, min_seam_valueVer = find_vertical_seam(t_dp[i][j-1][1])
            ver_value = min_seam_valueVer + t_dp[i][j-1][0]
            if hor_value < ver_value:
                t_dp[i][j] = (hor_value,delete_seam(t_dp[i-1][j][1],seamHor,isHorizontal=True),delete_seam(t_dp[i-1][j][2],seamHor,isHorizontal=True))
                # path_table[i][j] = 1
            else:
                t_dp[i][j] = (ver_value,delete_seam(t_dp[i][j-1][1],seamVer,isHorizontal=False),delete_seam(t_dp[i][j-1][2],seamVer,isHorizontal=False))
                # path_table[i][j] = 0

    return t_dp[r][c][2]


def showMin_X_Seams(f_image,e_image,number_of_seams, isHorizontal = True):

    if not isHorizontal:
        e_image = e_image.T
        f_image = np.rollaxis(f_image,1,0)
    out_image = np.copy(f_image)
    out_e_image = np.copy(e_image)

    seam_dp_table, path_table = get_DPtable_PathTable(e_image)
    final_column = list(seam_dp_table[:, e_image.shape[1] - 1])
    for i in range(len(final_column)):
        final_column[i] = (final_column[i], i)
    final_column.sort(key=lambda x: x[0])
    final_column = final_column[:number_of_seams]

    for k in range(number_of_seams):
        seam_path = construct_path_horizontal(path_table, final_column[k][1])
        for pixel in seam_path:
            out_image[pixel[0]][pixel[1]] = np.array([0,0,255])
            out_e_image[pixel[0]][pixel[1]] = 255

    if not isHorizontal:
        out_image = np.rollaxis(out_image,1,0)
        out_e_image = out_e_image.T
        e_image = e_image.T
        f_image = np.rollaxis(f_image,1,0)

    return out_image, out_e_image



def insert_seams_horizontal(f_image,e_image,number_of_seams):

    # print f_image.shape,e_image.shape

    height, length = e_image.shape
    out_image_prev = np.zeros((f_image.shape[0]+number_of_seams,f_image.shape[1],3),dtype=np.uint8)
    out_energy_image_prev = np.zeros((e_image.shape[0] + number_of_seams, e_image.shape[1]),dtype=np.uint8)

    out_image_prev[:f_image.shape[0],:,:] = np.copy(f_image)
    out_energy_image_prev[:e_image.shape[0],:] = np.copy(e_image)

    out_image_next = np.copy(out_image_prev)
    out_energy_image_next = np.copy(out_energy_image_prev)


    seam_dp_table, path_table = get_DPtable_PathTable(e_image)
    final_column = list(seam_dp_table[:, e_image.shape[1] - 1])
    for i in range(len(final_column)):
        final_column[i] = (final_column[i],i)
    final_column.sort(key = lambda x : x[0])
    final_column = final_column[:number_of_seams]
    final_column.sort(key = lambda x : x[1],reverse=True)
    # print final_column
    count = 0
    for k in range(number_of_seams):
        count = count + 1
        print(count),
        seam_path = construct_path_horizontal(path_table,final_column[k][1])
        # print seam_path
        for i in range(length):
            add_index = seam_path[i][0]
            # print add_index,
            for j in range(height-1+k,-1,-1):
                if j > add_index:
                    out_image_next[j+1][i] = out_image_prev[j][i]
                    out_energy_image_next[j+1][i] = out_energy_image_prev[j][i]
                    # print j,i
                    # print out_energy_image[j][i]
                    # print out_image[j][i]

                if j == add_index:

                    if j==height-1+k:
                        out_image_next[j+1][i] = out_image_prev[j][i]
                        out_energy_image_next[j+1][i] = out_energy_image_prev[j][i]
                    else:
                        out_image_next[j + 1][i] = (out_image_prev[j][i])/2 + (out_image_prev[j + 1][i]) / 2
                        out_energy_image_next[j + 1][i] = (out_energy_image_prev[j][i])/2 + (out_energy_image_prev[j + 1][i]) / 2


                if j <= add_index:
                    out_image_next[j][i] = out_image_prev[j][i]
                    out_energy_image_next[j][i] = out_energy_image_prev[j][i]

            # break

        out_image_prev = np.copy(out_image_next)
        out_energy_image_prev = np.copy(out_energy_image_next)
        # print out_energy_image_next

    return out_image_next,out_energy_image_next


def insert_seams_vertical(f_image,e_image,number_of_seams):
    f_image = np.rollaxis(f_image,1,0)
    e_image_T = e_image.T
    output_f_image_T, output_e_imagee_T = insert_seams_horizontal(f_image, e_image_T, number_of_seams)
    output_f_image_T = np.rollaxis(output_f_image_T,1,0)
    return output_f_image_T, output_e_imagee_T.T