import numpy as np
import cv2

import matching
import seamCarve


def perform_seam_carving(orig_input_img,energy_image, lengthChange=0, heightChange=0):
    orig_height, orig_length, _ = orig_input_img.shape

    required_height = orig_height + heightChange
    required_length = orig_length + lengthChange

    color, gray = seamCarve.showMin_X_Seams(orig_input_img, energy_image, orig_height, isHorizontal=True)
    # color,gray = seamCarve.showMin_X_Seams(orig_input_img,energy_image,1,isHorizontal=True)
    cv2.imshow("Best Seams", color)
    final_image = np.copy(orig_input_img)

    print("required length= " + str(required_length) + " required height= " + str(required_height))

    cur_height = orig_height
    cur_length = orig_length

    if required_length >= orig_length and required_height < orig_height:
        count = 0
        print ("Reducing Image Height. " + str(orig_height - required_height) + " operations required")
        while (required_height < cur_height):
            cur_height = cur_height - 1
            min_energy_seam, _ = seamCarve.find_horizontal_seam(energy_image)
            # print min_energy_seam
            energy_image = seamCarve.delete_seam(energy_image, min_energy_seam, isHorizontal=True)
            final_image = seamCarve.delete_seam(final_image, min_energy_seam, isHorizontal=True)
            count = count + 1
            print (count),

    if required_length < orig_length and required_height >= orig_height:
        count = 0
        print ("Reducing Image Length. " + str(orig_length - required_length) + " operations required")
        while (required_length < cur_length):
            cur_length = cur_length - 1
            min_energy_seam, _ = seamCarve.find_vertical_seam(energy_image)
            # print(min_energy_seam)
            energy_image = seamCarve.delete_seam(energy_image, min_energy_seam, isHorizontal=False)
            final_image = seamCarve.delete_seam(final_image, min_energy_seam, isHorizontal=False)
            count = count + 1
            print (count),

    if required_length < orig_length and required_height < orig_height:
        # Requires total of ((length_dif + 1) * (height_dif + 1))*2 carving operations;
        print("Reducing both length and height in optimal order")
        print (
            "Requires " + str(2 * ((orig_length - required_length) * (orig_height - required_height))) + " operations")
        final_image = seamCarve.get_optimal_order_image(energy_image, final_image, required_length, required_height)

    if required_length > orig_length or required_height > orig_height:
        excess_len_req = required_length - orig_length
        excess_hei_req = required_height - orig_height
        batch_size = 20

        if required_length > orig_length:
            length_diff = required_length - orig_length
            print ("Increasing Image Length. " + str(length_diff) + " operations required")
            while (length_diff > batch_size):
                final_image, energy_image = seamCarve.insert_seams_vertical(final_image, energy_image, batch_size)
                length_diff = length_diff - batch_size
            if length_diff > 0:
                final_image, energy_image = seamCarve.insert_seams_vertical(final_image, energy_image, length_diff)

        if required_height > orig_height:
            height_diff = required_height - orig_height
            print ("Increasing Image Height. " + str(height_diff) + " operations required")
            while (height_diff > batch_size):
                final_image, energy_image = seamCarve.insert_seams_horizontal(final_image, energy_image, batch_size)
                height_diff = height_diff - batch_size
            if height_diff > 0:
                final_image, energy_image = seamCarve.insert_seams_horizontal(final_image, energy_image, height_diff)

    return final_image, energy_image


def main():
    image_height = 500

    # img_path = "./Images/Waterfall.jpg"
    img_path = "./template_ims/ballonsample.jpg"


    orig_input_img = cv2.imread(img_path)
    r = (image_height) * 1.0 / orig_input_img.shape[0]
    dim = (int(orig_input_img.shape[1] * r), image_height)
    orig_input_img = cv2.resize(orig_input_img, dim, interpolation=cv2.INTER_AREA)

    orig_height, orig_length, _ = orig_input_img.shape
    print("length= " + str(orig_length) + " height= " + str(orig_height))

    orig_gray_img = cv2.cvtColor(orig_input_img, cv2.COLOR_BGR2GRAY)
    energy_image = seamCarve.gradient_filter(orig_gray_img)

    for j in range(energy_image.shape[1]):
        energy_image[0][j] = 100
        energy_image[energy_image.shape[0] - 1][j] = 100

    for j in range(energy_image.shape[0]):
        energy_image[j][0] = 100
        energy_image[j][energy_image.shape[1] - 1] = 100

    print("Do you want to \n(1) Remove objects from the image?\n(2) Resize Image\n(3) Remove Template\n ")
    answer = ''
    while (answer != '1' and answer != '2' and answer != '3'):
        print ("Enter 1,2 or 3")
        answer = raw_input()


    if answer == '1':

        img_copy_list = [np.copy(orig_input_img)]
        inProcessList = [False]
        energy_image_list = [energy_image]
        max_min_x = [0, 1000]
        max_min_y = [0, 1000]

        def selectRegMouse(event, x, y, flags, param):

            temp = x
            x = y
            y = temp
            inProcess = inProcessList[0]
            img_copy = img_copy_list[0]
            energy_image = energy_image_list[0]

            if event == cv2.EVENT_LBUTTONDOWN:
                inProcessList[0] = True
            elif event == cv2.EVENT_MOUSEMOVE:
                if inProcess:
                    if x > 3 and y > 3 and x < img_copy.shape[0] - 4 and y < img_copy.shape[1] - 4:
                        if x + 4 > max_min_x[0]:
                            max_min_x[0] = x+4
                        if x - 3 < max_min_x[1]:
                            max_min_x[1] = x-3
                        if y + 4 > max_min_y[0]:
                            max_min_y[0] = y+4
                        if y - 3 < max_min_y[1]:
                            max_min_y[1] = y-3
                        img_copy[x - 3:x + 4, y - 3:y + 4] = [0, 0, 255]
                        energy_image[x - 3:x + 4, y - 3:y + 4] = -500
            elif event == cv2.EVENT_LBUTTONUP:
                inProcessList[0] = False

        cv2.namedWindow("Region Selection for Removal")
        cv2.setMouseCallback("Region Selection for Removal", selectRegMouse)
        while True:
            cv2.imshow("Region Selection for Removal", img_copy_list[0])
            if cv2.waitKey(1) & 0xFF == 13:
                break
        cv2.destroyWindow("Region Selection for Removal")
        print("Region Selected")

        xDiff = max_min_x[0] - max_min_x[1]
        yDiff = max_min_y[0] - max_min_y[1]

        if xDiff <= yDiff:
            final_image, energy_image = perform_seam_carving(orig_input_img,energy_image,heightChange=-1*xDiff)
        else:
            final_image, energy_image = perform_seam_carving(orig_input_img, energy_image, lengthChange=-1 * yDiff)

    elif answer =='2':
        print("Enter desired change in height")
        height_change = int(raw_input())
        print("Enter desired change in length")
        length_change = int(raw_input())
        final_image, energy_image = perform_seam_carving(orig_input_img, energy_image, length_change, height_change)

    elif answer =='3':
        template = cv2.imread('./template_ims/tempsample.jpg')
        dim = (int(template.shape[1] * r), int(template.shape[0] * r))
        template = cv2.resize(template, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Template",template)
        match = matching.find_exact_match_coordinates(orig_input_img,template)
        yTopLeft = match[0]
        xTopLeft = match[1]
        width = match[2]
        height = match[3]
        copy_orig = np.copy(orig_input_img)
        cv2.rectangle(copy_orig, (yTopLeft, xTopLeft), (yTopLeft + width, xTopLeft + height),
                      (0, 0, 255))
        cv2.imshow("Exact Match",copy_orig)
        energy_image[xTopLeft:xTopLeft+height,yTopLeft:yTopLeft+width] = -500

        if height <= width:
            final_image, energy_image = perform_seam_carving(orig_input_img,energy_image,heightChange=-1*height)
        else:
            final_image, energy_image = perform_seam_carving(orig_input_img, energy_image, lengthChange=-1 * width)

    for i in range(energy_image.shape[0]):
        for j in range(energy_image.shape[1]):
            if energy_image[i][j] < 0:
                energy_image[i][j] = 0
    energy_image = energy_image.astype(np.uint8)

    cv2.imshow("Energy Image", energy_image)
    cv2.imshow("Original Image", orig_input_img)
    cv2.imshow("Final_Image", final_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()
