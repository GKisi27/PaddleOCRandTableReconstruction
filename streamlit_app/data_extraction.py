import os
import cv2
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from clean_dataframe import *
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont

# Load the custom font (adjust path if needed)
font_path = "./fonts/simfang.ttf" 
text_color = (255, 0, 0)    # Red

def reset_column(df):
    df_transposed = df.T
    df_transposed.reset_index(inplace=True)
    df_transposed.columns = range(df_transposed.shape[1])
    df_retransposed = df_transposed.T
    return df_retransposed

def get_images(folder_path):
    image_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
            image_files.append(filename)
    return image_files

def create_folder(filepath):
    # Create the directory
    try:
        os.mkdir(filepath)
        print(f"Folder created successfully!: {filepath}")
    except FileExistsError:
        print(f"Folder already exists!: {filepath}")

def intersection(box_1, box_2):
    return [box_2[0], box_1[1],box_2[2], box_1[3]]

def iou(box_1, box_2):

    x_1 = max(box_1[0], box_2[0])
    y_1 = max(box_1[1], box_2[1])
    x_2 = min(box_1[2], box_2[2])
    y_2 = min(box_1[3], box_2[3])

    inter = abs(max((x_2 - x_1, 0)) * max((y_2 - y_1), 0))
    if inter == 0:
            return 0

    box_1_area = abs((box_1[2] - box_1[0]) * (box_1[3] - box_1[1]))
    box_2_area = abs((box_2[2] - box_2[0]) * (box_2[3] - box_2[1]))

    return inter / float(box_1_area + box_2_area - inter)


def extract_data(ocr, img_path, scale, count = 1):
    if count<10:
        count="0"+str(count)
    font_size = 15*scale
    font = ImageFont.truetype(font_path, font_size)
    pil_image = Image.open(img_path).convert('RGB')
    cv2_image = cv2.imread(img_path)
    image_height = cv2_image.shape[0]
    image_width = cv2_image.shape[1]
    result = ocr.ocr(img_path, cls=False)[0]

    boxes = [line[0] for line in result]
    texts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]

    ## Save image as result with OCR detection values
    for box, text in zip(boxes,texts):
        
        rectangle_coords =[(int(box[0][0]),int(box[0][1])), (int(box[2][0]),int(box[2][1]))]
        draw = ImageDraw.Draw(pil_image)

        bbox = draw.textbbox((0, 0), text, font=font)
        text_width, text_height = bbox[2] - bbox[0], bbox[3] - bbox[1]
        rect_width = rectangle_coords[1][0] - rectangle_coords[0][0]

        # Center the text horizontally relative to the rectangle
        text_x = rectangle_coords[1][0] - text_width
        text_y = rectangle_coords[0][1] - text_height
        text_position = (text_x, text_y)

        # Draw the text on the image
        draw.text(text_position, text, font=font, fill='red')
        # # Draw the rectangle
        draw.rectangle(rectangle_coords, outline='green', width=1)
        # Convert image back to OpenCV format
        opencv_img = np.array(pil_image)


    ##Reconstruction
    im = cv2_image.copy()
    horiz_boxes = []
    vert_boxes = []

    for box in boxes:
        x_h, x_v = 0,int(box[0][0])
        y_h, y_v = int(box[0][1]),0
        width_h,width_v = image_width, int(box[2][0]-box[0][0])
        height_h,height_v = int(box[2][1]-box[0][1]),image_height

        horiz_boxes.append([x_h,y_h,x_h+width_h,y_h+height_h])
        vert_boxes.append([x_v,y_v,x_v+width_v,y_v+height_v])

        cv2.rectangle(im,(x_h,y_h), (x_h+width_h,y_h+height_h),(0,0,255),1)
        cv2.rectangle(im,(x_v,y_v), (x_v+width_v,y_v+height_v),(0,255,0),1)
        

    #Non-Max Suppression ## Keeping the bounding box with highest confidence score.
    horiz_out = tf.image.non_max_suppression(
            horiz_boxes,
            scores,
            max_output_size = 1200,
            iou_threshold=0.02,
            score_threshold=float('-inf'),
            name=None
    )

    horiz_lines = np.sort(np.array(horiz_out))

    im_nms = cv2_image.copy()

    for val in horiz_lines:
        cv2.rectangle(im_nms, (int(horiz_boxes[val][0]),int(horiz_boxes[val][1])), (int(horiz_boxes[val][2]),int(horiz_boxes[val][3])),(0,0,255),1)

    vert_out = tf.image.non_max_suppression(
            vert_boxes,
            scores,
            max_output_size = 1000,
            iou_threshold=0.5,
            score_threshold=float('-inf'),
            name=None
    )

    vert_lines = np.sort(np.array(vert_out))

    for val in vert_lines:
        cv2.rectangle(im_nms, (int(vert_boxes[val][0]),int(vert_boxes[val][1])), (int(vert_boxes[val][2]),int(vert_boxes[val][3])),(255,0,0),1)

    #Convert to CSV
    out_array = [["" for i in range(len(vert_lines))] for j in range(len(horiz_lines))]

    unordered_boxes = []

    for i in vert_lines:
        unordered_boxes.append(vert_boxes[i][0])
        
    ordered_boxes = np.argsort(unordered_boxes)

    for i in range(len(horiz_lines)):
        for j in range(len(vert_lines)):
            resultant = intersection(horiz_boxes[horiz_lines[i]], vert_boxes[vert_lines[ordered_boxes[j]]] )

            for b in range(len(boxes)):
                the_box = [boxes[b][0][0],boxes[b][0][1],boxes[b][2][0],boxes[b][2][1]]
                if(iou(resultant,the_box)>0.1):
                    out_array[i][j] = texts[b]
                    
    out_array = np.array(out_array)

    df = pd.DataFrame(out_array)
    return df