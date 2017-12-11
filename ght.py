import numpy as np
import cv2


def buildRefTable(img):
    table = [[0 for x in range(1)] for y in range(90)]  # creating a empty list
    img_center = [img.shape[0]/2, img.shape[1]/2] # r will be calculated corresponding to this point

    filter_size = 3
    for x in range(img.shape[0]-(filter_size-1)):
        for y in range(img.shape[1]-(filter_size-1)):
            if img[x,y] != 0:
                theta, r = findAngleDistance(x,y,img_center)
                if r != 0:
                    table[np.absolute(theta)].append(r)

    for i in range(len(table)): table[i].pop(0)
    return table


def findAngleDistance(x1,y1,img_center):
    x2, y2 = img_center[0], img_center[1]
    r = [(x2-x1),(y2-y1)]
    if (x2-x1 != 0):
        return [int(np.rad2deg(np.arctan((y2-y1)/(x2-x1)))), r]
    else:
        return [0,0]


def findMaxima(acc):

    ridx,cidx = np.unravel_index(acc.argmax(),acc.shape)
    return [acc[ridx,cidx],ridx,cidx]


def matchTable(im, table):

    m, n = im.shape
    acc = np.zeros((m+50,n+50)) # Extra space as voted points for shapes can be outside the image

    def findGradient(x,y):
        if (x != 0):
            return int(np.rad2deg(np.arctan(y/x)))
        else:
            return 0

    for x in range(1,im.shape[0]):
        for y in range(im.shape[1]):

            if im[x,y] != 0: # boundary point
                theta = findGradient(x,y)
                vectors = table[theta]
                for vector in vectors:
                    acc[vector[0]+x, vector[1]+y]+=1
    return acc

def main(template_names,actual_image_name):

    im = cv2.imread(actual_image_name,0)
    for img in template_names:
        refim = cv2.imread(img,0)

        table = buildRefTable(refim)
        acc = matchTable(im, table)
        val, ridx, cidx = findMaxima(acc)


        # code for drawing bounding-box in original image at the found location

        # find the half-width and height of template
        hheight = np.floor(refim.shape[0] / 2) + 1
        hwidth = np.floor(refim.shape[1] / 2) + 1

        # find coordinates of the box
        rstart = int(max(ridx - hheight, 1))
        rend = int(min(ridx + hheight, im.shape[0] - 1))
        cstart = int(max(cidx - hwidth, 1))
        cend = int(min(cidx + hwidth, im.shape[1] - 1))

        # draw the box
        im[rstart:rend, cstart] = 255
        im[rstart:rend, cend] = 255

        im[rstart, cstart:cend] = 255
        im[rend, cstart:cend] = 255

        # show the image
        cv2.imshow("Reference Image",refim)
        cv2.imshow("Image",im)


if __name__ == '__main__':
    template_images = ['./GHT/templates/Input1Ref.png', './GHT/templates/Input2Ref.png']
    search_in = './GHT/actual/Input1.png'
    main(template_images,search_in)
    cv2.waitKey()
