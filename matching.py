import cv2
import numpy as np
import math
# import imutils


def do_matching(image, template):
    fft_image = np.fft.fft2(image)
    fft_template = np.fft.fft2(template, s=image.shape)  # crop the inputs to the size of image
    finding_corelation = np.multiply(fft_image, np.conj(fft_template))
    inverse_transform = np.fft.ifft2(
        np.divide(finding_corelation, np.absolute(finding_corelation)))  # take the inverse transform
    real_component = np.real(inverse_transform)
    return np.argmax(real_component) / image.shape[1], np.argmax(real_component) % image.shape[
        1], real_component  # return (x,y) of the concerned point


# actual_image to create the box around
def rotate_matching(actual_image, actual_template, flag):
    gray_image = cv2.cvtColor(actual_image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(actual_template, cv2.COLOR_BGR2GRAY)
    number_of_rotations = 0
    rotated_points = []
    for some_angle in np.arange(-90, 90, 15):
        temp = np.copy(actual_image)
        rotated_template = imutils.rotate_bound(gray_template, some_angle)
        top_left_x, top_left_y, value = do_matching(gray_image, rotated_template)
        # bottom_right_x = x + scaling_factor*gray_template.shape[0]
        # bottom_right_y = y + scaling_factor*gray_template.shape[1]
        # cv2.rectangle(actual_image, (top_left_x, top_left_y), ((int)bottom_right_x, (int)bottom_right_y), (0,0,255))
        rotated_points.append(
            (top_left_y, top_left_x, (int)(rotated_template.shape[1]), (int)(rotated_template.shape[0]), value))
        cv2.circle(temp, (top_left_y, top_left_x), 30, (0, 0, 0))
        cv2.circle(temp, (top_left_y, top_left_x), 3, (0, 0, 255))
        # cv2.rectangle(temp, )
        cv2.imshow('Rotated Template: ' + str(number_of_rotations), imutils.rotate_bound(actual_template, some_angle))
        cv2.imshow('Actual Image (ROTATION) ' + str(number_of_rotations), temp)
        number_of_rotations += 1
        if flag == 1:
            cv2.waitKey(0)
    return rotated_points


def scaled_matching(actual_image, actual_template, flag):
    gray_image = cv2.cvtColor(actual_image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(actual_template, cv2.COLOR_BGR2GRAY)
    number_of_scalings = 0
    scaled_points = []
    for scaling_factor in np.arange(0.4, 2.4, 0.2):
        temp = np.copy(actual_image)
        scaled_template = cv2.resize(gray_template, (0, 0), fx=scaling_factor, fy=scaling_factor)
        top_left_x, top_left_y, value = do_matching(gray_image, scaled_template)
        scaled_points.append((top_left_y, top_left_x, scaling_factor * gray_template.shape[1],
                              scaling_factor * gray_template.shape[0], value))
        bottom_right_x = top_left_x + (int)(scaling_factor * gray_template.shape[0])
        bottom_right_y = top_left_y + (int)(scaling_factor * gray_template.shape[1])
        cv2.rectangle(temp, (top_left_y, top_left_x), ((int)(bottom_right_y), (int)(bottom_right_x)), (0, 0, 255))
        # cv2.circle(temp, (top_left_y, top_left_x), 5, (0, 255, 0))
        cv2.imshow('Scaled Template: ' + str(number_of_scalings),
                   cv2.resize(actual_template, (0, 0), fx=scaling_factor, fy=scaling_factor))
        cv2.imshow('Actual Image (SCALE)' + str(number_of_scalings), temp)
        number_of_scalings += 1
        if flag == 1:
            cv2.waitKey(0)
    return scaled_points


def multiple_matching(actual_image, actual_template, flag):
    gray_image = cv2.cvtColor(actual_image, cv2.COLOR_BGR2GRAY)
    gray_template = cv2.cvtColor(actual_template, cv2.COLOR_BGR2GRAY)
    _, _, ift = do_matching(gray_image, gray_template)
    flat = ift.flatten()
    sorted_flat = np.sort(flat)
    index_sorted = np.argsort(flat)
    all_matchings = []
    for i in range(1, 20):
        x = index_sorted[-i] / actual_image.shape[1]
        y = index_sorted[-i] % actual_image.shape[1]
        value = index_sorted[-i]
        all_matchings.append((y, x, (int)(actual_template.shape[1]), (int)(actual_template.shape[0]), value))
        cv2.circle(actual_image, (y, x), 5, (255, 0, 0))
        cv2.rectangle(actual_image, (y, x), (y + (int)(actual_template.shape[1]), x + (int)(actual_template.shape[0])),
                      (0, 0, 255))
        cv2.imshow("Result", actual_image)
        if flag == 1:
            cv2.waitKey(0)
    if flag == 1:
        cv2.waitKey(0)
    return all_matchings


def find_exact_match_coordinates(image,template):
    if len(image.shape) == 3:
        grayImage = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    else:
        grayImage = np.copy(image)
    if len(template.shape) == 3:
        grayTemplate = cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
    else:
        grayTemplate = np.copy(template)
    xTopLeft, yTopLeft, _ = do_matching(grayImage, grayTemplate)
    return (yTopLeft,xTopLeft,template.shape[1],template.shape[0])


def main():
    image = cv2.imread('./template_ims/ballonsample.jpg')
    template = cv2.imread('./template_ims/tempsample.jpg')
    image = cv2.resize(image,None,fx=0.5,fy=0.5)
    template = cv2.resize(template, None, fx=0.5, fy=0.5)

    # image = cv2.imread('./images/hotairballoon.png')
    # template = cv2.imread('./images/hotairtemplate.jpg')

    # image = cv2.imread('./images/CardsPng/clubs_test.png')
    # template = cv2.imread('./images/CardsPng/clubs3.jpg')

    # image = cv2.imread('./images/hotairballoon.png')
    # template = cv2.imread('./images/templatehotair.png')

    # image = cv2.imread('./images/multipleballoons.jpg')
    # template = cv2.imread('./images/balloontemplate.jpg')

    # image = cv2.imread('./images/balloonrotation.jpg')
    # template = cv2.imread('./images/balloonrotationtemplate.jpg')

    # image = cv2.imread('./images/scaling_matching.jpg')
    # template = cv2.imread('./images/scaling_matching_template.jpg')

    cv2.imshow('main',image)
    cv2.imshow('template',template)

    grayim = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    graytemp = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    # scaled_points = scaled_matching(np.copy(image), template, 1)
    # best_scaled = sorted(scaled_points, key=lambda x: x[4], reverse = True) #sorted them on the based of their values
    #
    # rotated_points = rotate_matching(np.copy(image), template, 1)
    # # best_rotated = sorted(rotated_points, key=lambda x: x[4], reverse = True)
    #
    # matched_points = multiple_matching(np.copy(image), template, 1)
    # # best_matched = sorted(matched_points, key=lambda x: x[4], reverse = True)

    xTopLeft, yTopLeft, _ = do_matching(grayim, graytemp)
    # cir_radius = int(math.sqrt(template.shape[0]**2 + template.shape[1]**2)/2)
    # cv2.circle(image, (yCentre, xCentre), cir_radius, (0, 255, 0))
    cv2.rectangle(image, (yTopLeft, xTopLeft), (yTopLeft + template.shape[1], xTopLeft + template.shape[0]),
                  (0, 0, 255))
    cv2.imshow('Match', image)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()
