import cv2
import os
from constuct_lbp import lbp_calculated_pixel
from constuct_lbp import show_output
import numpy as np


def calculateLBPVector(im):

    img_ycrcb = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)

    img_ycrcb = cv2.split(img_ycrcb)
    img_yuma = img_ycrcb[0]
    img_cr = img_ycrcb[1]
    img_cb = img_ycrcb[2]


    height, width, channel = im.shape

    img_lbp_yuma = np.zeros((height, width, 3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp_yuma[i, j] = lbp_calculated_pixel(img_yuma, i, j)

    img_lbp_cr = np.zeros((height, width, 3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp_cr[i, j] = lbp_calculated_pixel(img_cr, i, j)

    img_lbp_cb = np.zeros((height, width, 3), np.uint8)
    for i in range(0, height):
        for j in range(0, width):
            img_lbp_cb[i, j] = lbp_calculated_pixel(img_cb, i, j)

    hist_lbp_yuma = cv2.calcHist([img_lbp_yuma], [0], None, [256], [0, 256])
    hist_lbp_cr = cv2.calcHist([img_lbp_cr], [0], None, [256], [0, 256])
    hist_lbp_cb = cv2.calcHist([img_lbp_cb], [0], None, [256], [0, 256])

    final_hist = np.concatenate((hist_lbp_yuma, hist_lbp_cr, hist_lbp_cb))
    return final_hist



'''output_list1 = []
output_list2 = []
output_list3 = []


#print(f"Histogram cR: {hist_lbp_cr}")
#print(f"Histogram cB: {hist_lbp_cb}")



output_list1.append({
    "img": img_yuma,
    "xlabel": "",
    "ylabel": "",
    "xtick": [],
    "ytick": [],
    "title": "Yuma Channel",
    "type": "gray"
})
output_list1.append({
    "img": img_lbp_yuma,
    "xlabel": "",
    "ylabel": "",
    "xtick": [],
    "ytick": [],
    "title": "Yuma LBP",
    "type": "gray"
})
output_list1.append({
    "img": hist_lbp_yuma,
    "xlabel": "Bins",
    "ylabel": "Number of pixels",
    "xtick": None,
    "ytick": None,
    "title": "Yuma Histogram",
    "type": "histogram"
})



output_list2.append({
    "img": img_cr,
    "xlabel": "",
    "ylabel": "",
    "xtick": [],
    "ytick": [],
    "title": "cR Channel",
    "type": "gray"
})
output_list2.append({
    "img": img_lbp_cr,
    "xlabel": "",
    "ylabel": "",
    "xtick": [],
    "ytick": [],
    "title": "cR LBP",
    "type": "gray"
})

output_list2.append({
    "img": hist_lbp_cr,
    "xlabel": "Bins",
    "ylabel": "Number of pixels",
    "xtick": None,
    "ytick": None,
    "title": "Histogram(LBP)",
    "type": "histogram"
})

output_list3.append({
    "img": img_cb,
    "xlabel": "",
    "ylabel": "",
    "xtick": [],
    "ytick": [],
    "title": "cB Channel",
    "type": "gray"
})
output_list3.append({
    "img": img_lbp_cb,
    "xlabel": "",
    "ylabel": "",
    "xtick": [],
    "ytick": [],
    "title": "cB LBP",
    "type": "gray"
})
output_list3.append({
    "img": hist_lbp_cb,
    "xlabel": "Bins",
    "ylabel": "Number of pixels",
    "xtick": None,
    "ytick": None,
    "title": "cB Histogram",
    "type": "histogram"
})

#show_output(output_list1)
#show_output(output_list2)
#show_output(output_list3)

#cv2.waitKey(0) '''
