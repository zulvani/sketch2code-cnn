from flask import Flask, jsonify, request
import cv2
import numpy as np
import pandas as pd
from flask_cors import CORS
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image
import keras
from ultralytics import YOLO

import os
from datetime import datetime
import pytesseract
from collections import defaultdict

app = Flask(__name__)
CORS(app)  # Mengaktifkan CORS untuk semua endpoint

UPLOAD_FOLDER = '/tmp/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET"])
def hello():
    return jsonify({"message": "Hello, World!"})

@app.route('/api/detect/v3', methods=['POST'])
def detect_object_v3():
    fullPathInputImageFile = ''
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file to temp folder
    filename = datetime.now().strftime('%Y%m%d%H%M%S_') + file.filename
    fullPathInputImageFile = os.path.join(UPLOAD_FOLDER, filename)
    file.save(fullPathInputImageFile)

    method = request.args.get('method')
    if method is None or method == '':
        method = 'contour-detection'

    image = cv2.imread(fullPathInputImageFile, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return jsonify({'error': 'Failed to load the input image. Please check the file path.'}), 400

    # extract word boxes using OCR
    boxes = get_word_boxes(image)
    groups = group_by_smart_proximity(boxes, gap_multiplier=1.2)

    merged_results = []
    for group in groups:
        merged_box = merge_group_bboxes(group)
        if merged_box:
            merged_results.append(merged_box)

    if method == 'contour-detection':
        # Convert the image to binary image (black and white)
        _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

        # sharpening the objects
        kernel = np.ones((3, 3), np.uint8)  # 5x5 square kernel
        image = cv2.dilate(image, kernel, iterations=1)

        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxs = []    
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            diff = 1

            # Calculate x2 and y2 for bounding box
            ax1, ay1, ax2, ay2 = x + diff , y + diff, x + w - diff, y + h - diff
            
            # Check if this bounding box matches any OCR box
            found = False
            for box in merged_results:
                bbox = box['bbox']
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                if (ax1 < x1 and ax2 > x2 and ay1 < y1 and ay2 > y2):
                    box['inside'] = True
                    break

                di = 3
                if (ax1 >= x1 - di and ay1 >= y1 - di and ax2 <= x2 + di and ay2 <= y2 + di):
                    found = True
                    break
            
            if not found:  
                res = {'x1': ax1, 'y1': ay1, 'x2': ax2, 'y2': ay2, 'obj' : None, 'w': w, 'h': h}
                bounding_boxs.append(res)
        
        for box in merged_results:
            if not box['inside']:
                bbox = box['bbox']
                x1, y1, x2, y2, w, h = bbox[0], bbox[1], bbox[2], bbox[3], x2 - x1, y2 - y1

                objClass = 'label'
                if 'http' in box['text'] or 'https' in box['text'] or '.com' in box['text'] or 'www' in box['text']:
                    objClass = 'hyperlink'

                res = {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'obj' : {'text': box['text'], 'objClass': objClass, 'word_count': box['word_count']}, 'w': w, 'h': h
                }
                bounding_boxs.append(res)

        for box in bounding_boxs:
            for ocr_box in merged_results:
                bbox = ocr_box['bbox']
                if box['x1'] < bbox[0] and box['x2'] > bbox[2] and box['y1'] < bbox[1] and box['y2'] > bbox[3]:
                    text = ocr_box['text']
                    if box['obj'] is None:
                        box['obj'] = {'text': text}
                    break
    
    # Currently YOLOv11 is not supported in this version
    elif method == 'yolov11':
        return demo()
    
    return jsonify(bounding_boxs)


@app.route('/api/detect/v2', methods=['GET'])
def detect_object_v2():
    fullPathInputImageFile = request.args.get('in')
    
    method = request.args.get('method')
    if method is None or method == '':
        method = 'contour-detection'

    image = cv2.imread(fullPathInputImageFile, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return jsonify({'error': 'Failed to load the input image. Please check the file path.'}), 400

    # extract word boxes using OCR
    boxes = get_word_boxes(image)
    groups = group_by_smart_proximity(boxes, gap_multiplier=1.2)

    merged_results = []
    for group in groups:
        merged_box = merge_group_bboxes(group)
        if merged_box:
            merged_results.append(merged_box)

    if method == 'contour-detection':
        # Convert the image to binary image (black and white)
        _, image = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY_INV)

        # sharpening the objects
        kernel = np.ones((3, 3), np.uint8)  # 5x5 square kernel
        image = cv2.dilate(image, kernel, iterations=1)

        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxs = []    
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            diff = 1

            # Calculate x2 and y2 for bounding box
            ax1, ay1, ax2, ay2 = x + diff , y + diff, x + w - diff, y + h - diff
            
            # Check if this bounding box matches any OCR box
            found = False
            for box in merged_results:
                bbox = box['bbox']
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                if (ax1 < x1 and ax2 > x2 and ay1 < y1 and ay2 > y2):
                    box['inside'] = True
                    break

                di = 3
                if (ax1 >= x1 - di and ay1 >= y1 - di and ax2 <= x2 + di and ay2 <= y2 + di):
                    found = True
                    break
            
            if not found:  
                res = {'x1': ax1, 'y1': ay1, 'x2': ax2, 'y2': ay2, 'obj' : None, 'w': w, 'h': h}
                bounding_boxs.append(res)
        
        for box in merged_results:
            if not box['inside']:
                bbox = box['bbox']
                x1, y1, x2, y2, w, h = bbox[0], bbox[1], bbox[2], bbox[3], x2 - x1, y2 - y1

                objClass = 'label'
                if 'http' in box['text'] or 'https' in box['text'] or '.com' in box['text'] or 'www' in box['text']:
                    objClass = 'hyperlink'

                res = {
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'obj' : {'text': box['text'], 'objClass': objClass, 'word_count': box['word_count']}, 'w': w, 'h': h
                }
                bounding_boxs.append(res)

        for box in bounding_boxs:
            for ocr_box in merged_results:
                bbox = ocr_box['bbox']
                if box['x1'] < bbox[0] and box['x2'] > bbox[2] and box['y1'] < bbox[1] and box['y2'] > bbox[3]:
                    text = ocr_box['text']
                    if box['obj'] is None:
                        box['obj'] = {'text': text}
                    break
    
    # Currently YOLOv11 is not supported in this version
    elif method == 'yolov11':
        return demo()
    
    return jsonify(bounding_boxs)

@app.route('/api/detect/v1', methods=['GET'])
def demo():
    fullPathInputImageFile = request.args.get('in')
    fullPathOutputImageFile = request.args.get('out')
    
    method = request.args.get('method')
    if method is None or method == '':
        method = 'contour-detection'

    citra = cv2.imread(fullPathInputImageFile, cv2.IMREAD_GRAYSCALE)
    if citra is None:
        return jsonify({'error': 'Failed to load the input image. Please check the file path.'}), 400

    citra = binarization(citra, 200)

    # Define a kernel for dilation
    kernel = np.ones((3, 3), np.uint8)  # 5x5 square kernel
    citra = cv2.dilate(citra, kernel, iterations=1)
    
    bounding_boxs = []

    if method == 'contour-detection':
        contours, hierarchy = cv2.findContours(citra, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output_image = cv2.cvtColor(citra, cv2.COLOR_GRAY2BGR)
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            bounding_boxs.append([x,y, x+w, y+h, w, h])

            cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

            top_left = (x, y)
            bottom_right = (x + w, y + h)

            # Draw the top-left corner coordinates
            cv2.putText(output_image, str(top_left), top_left, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (239, 245, 66), 1)

            # Draw the bottom-right corner coordinates
            cv2.putText(output_image, str(bottom_right), bottom_right, cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (239, 245, 66), 1)

            # Save the output image to the hard disk
            # output_image_path = base_path + 'output_ui.png'
            if fullPathOutputImageFile is not None and fullPathOutputImageFile != '':
                cv2.imwrite(fullPathOutputImageFile, output_image)
    elif method == 'yolov3':
        # best_model = YOLO('/Users/aguszulvani/privacy/MKOM/TESIS/EXPERIENCE-RESULT/YOLO/v3-T4GPU/weights/best.pt')
        best_model = YOLO('/Users/aguszulvani/privacy/MKOM/TESIS/EXPERIENCE-RESULT/YOLO/YOLOv11/weights/best.pt')
        results = best_model.predict(fullPathInputImageFile, imgsz=citra.shape[1], agnostic_nms=False)
        hasil = results[0]
        
        # Get the coordinates of the top-left and bottom-right positions
        for box in hasil.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  

            w = x2 - x1
            h = y2 - y1

            # Draw the bounding box on the image
            bounding_boxs.append([x1,y1, x2, y2, w, h])
            
    return jsonify(bounding_boxs)

@app.route('/api/classify/v2', methods=['POST'])
def classifyV2():
    predicted_class_name = 'undefined'

    image_width = 200
    image_height = 141 

    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save uploaded file to temp folder
    filename = datetime.now().strftime('%Y%m%d%H%M%S_object_') + file.filename
    object = os.path.join(UPLOAD_FOLDER, filename)
    file.save(object)

    model = request.args.get('model') 
    if model is None or model == '':
        model = 'sdn' 

    if model == 'yolov3':
        # YOLOv3 -----------
        print("Using YOLOv3 for classification-----------------")
        # best_model = YOLO('/Users/aguszulvani/privacy/MKOM/TESIS/EXPERIENCE-RESULT/YOLO/v3-T4GPU/weights/best.pt')
        best_model = YOLO('/Users/aguszulvani/privacy/MKOM/TESIS/EXPERIENCE-RESULT/YOLO/YOLOv11/weights/best.pt')
        yolo_v3_class_names = ['alert','bar-chart','checkbox', 'combobox', 'common-button',
                       'common-image-button','date-picker','hyperlink','icon-button', 
                       'image','image-card','input-file', 'input-free-text','input-number',
                       'input-password','key-value','label','line-chart', 'list','pie-chart', 
                       'radio-button','segmented-button','slider','switch', 
                       'table','textarea','time-picker'] 
         
        image_path = object
        image = Image.open(image_path).convert("L")
        results = best_model.predict(image_path, imgsz=(image_width, image_height))
        for result in results:
            boxes = result.boxes  
            if len(boxes.cls) > 0:
                predicted_class_index = int(boxes.cls[0])
                predicted_class_name = yolo_v3_class_names[predicted_class_index]
        return jsonify({'objectClass': predicted_class_name})
    else:
        # Sketch Deep Net -----------
        print("Using Sketch Deep Net for classification-----------------")
        class_names = ['input-free-text', 'radio-button', 'hyperlink', 'input-number', 
                       'time-picker', 'image-card', 'alert', 'pie-chart', 'checkbox', 
                       'label', 'slider', 'bar-chart', 'common-image-button', 'image', 
                       'combobox', 'key-value', 'textarea', 'date-picker', 'table', 'list', 
                       'segmented-button', 'common-button', 'input-password', 'switch', 
                       'line-chart', 'icon-button', 'input-file']
        #model = load_model("cnn_model_20250422-v2.keras")
        # model = load_model("cnn_model_2025_08_24.keras")
        # model = load_model("Sketch-DeepNet-e23-mPrecission-2026-01-29.keras")
        model = load_model("Sketch-DeepNet-e35-2026-01-29.keras")
        image_path = object
        image = Image.open(image_path).convert("L")
        image_array = np.array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]
        return jsonify({'objectClass': predicted_class_name})


@app.route('/api/classify', methods=['GET'])
def classify():
    predicted_class_name = 'undefined'

    image_width = 200
    image_height = 141 

    object = request.args.get('object') 
    model = request.args.get('model') 

    if model is None or model == '':
        model = 'sdn' 

    if model == 'yolov3':
        # YOLOv3 -----------
        print("Using YOLOv3 for classification-----------------")
        # best_model = YOLO('/Users/aguszulvani/privacy/MKOM/TESIS/EXPERIENCE-RESULT/YOLO/v3-T4GPU/weights/best.pt')
        best_model = YOLO('/Users/aguszulvani/privacy/MKOM/TESIS/EXPERIENCE-RESULT/YOLO/YOLOv11/weights/best.pt')
        yolo_v3_class_names = ['alert','bar-chart','checkbox', 'combobox', 'common-button',
                       'common-image-button','date-picker','hyperlink','icon-button', 
                       'image','image-card','input-file', 'input-free-text','input-number',
                       'input-password','key-value','label','line-chart', 'list','pie-chart', 
                       'radio-button','segmented-button','slider','switch', 
                       'table','textarea','time-picker'] 
         
        image_path = object
        image = Image.open(image_path).convert("L")
        results = best_model.predict(image_path, imgsz=(image_width, image_height))
        for result in results:
            boxes = result.boxes  
            if len(boxes.cls) > 0:
                predicted_class_index = int(boxes.cls[0])
                predicted_class_name = yolo_v3_class_names[predicted_class_index]
        return jsonify({'objectClass': predicted_class_name})
    else:
        # Sketch Deep Net -----------
        print("Using Sketch Deep Net for classification-----------------")
        class_names = ['input-free-text', 'radio-button', 'hyperlink', 'input-number', 
                       'time-picker', 'image-card', 'alert', 'pie-chart', 'checkbox', 
                       'label', 'slider', 'bar-chart', 'common-image-button', 'image', 
                       'combobox', 'key-value', 'textarea', 'date-picker', 'table', 'list', 
                       'segmented-button', 'common-button', 'input-password', 'switch', 
                       'line-chart', 'icon-button', 'input-file']
        #model = load_model("cnn_model_20250422-v2.keras")
        model = load_model("cnn_model_2025_08_24.keras")
        image_path = object
        image = Image.open(image_path).convert("L")
        image_array = np.array(image)
        image_array = image_array / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]
        return jsonify({'objectClass': predicted_class_name})

def binarization(citra, thresold):
    row = citra.shape[0]
    column = citra.shape[1]
    hasil = np.zeros((row, column), np.uint8)

    # Define a kernel for dilation
    # kernel = np.ones((1, 1), np.uint8)  # 5x5 square kernel
    # citra = cv2.dilate(citra, kernel, iterations=1)

    for i in range(row):
        for j in range(column):
            if citra[i, j] >= thresold:
                hasil[i, j] = 0
            else:
                hasil[i, j] = 255    
    return hasil 

def get_word_boxes(image):
    
    # Get word-level data from pytesseract
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    boxes = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 30:  # Filter low confidence
            text = data['text'][i].strip()
            if text:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                boxes.append({
                    'text': text,
                    'bbox': (x, y, x + w, y + h),  # (x1, y1, x2, y2)
                    'conf': data['conf'][i],
                    'line_num': data['line_num'][i],
                    'word_num': data['word_num'][i],
                    'char_width': w / len(text) if len(text) > 0 else w  # Average character width
                })
    
    return boxes

# Group words within a line based on spacing analysis
def group_words_in_line(line_words, gap_multiplier):
    if len(line_words) <= 1:
        return [line_words]
    
    # Calculate character-based spacing threshold
    char_widths = [word['char_width'] for word in line_words]
    avg_char_width = np.mean(char_widths)
    
    # Dynamic threshold based on average character width
    spacing_threshold = avg_char_width * gap_multiplier
    
    return _group_by_threshold(line_words, spacing_threshold)

def _group_by_threshold(line_words, threshold):
    if not line_words:
        return []
    
    groups = []
    current_group = [line_words[0]]
    
    for i in range(1, len(line_words)):
        prev_word = line_words[i-1]
        curr_word = line_words[i]
        
        gap = curr_word['bbox'][0] - prev_word['bbox'][2]
        
        if gap <= threshold:
            current_group.append(curr_word)
        else:
            groups.append(current_group)
            current_group = [curr_word]
    
    if current_group:
        groups.append(current_group)
    
    return groups

def _group_by_lines(boxes, vertical_tolerance=10):
    lines = defaultdict(list)
    
    for box in boxes:
        y_center = (box['bbox'][1] + box['bbox'][3]) // 2
        
        # Find existing line or create new one
        line_key = None
        for existing_y in lines.keys():
            if abs(y_center - existing_y) <= vertical_tolerance:
                line_key = existing_y
                break
        
        if line_key is None:
            line_key = y_center
        
        lines[line_key].append(box)
    
    return list(lines.values())

def group_by_smart_proximity(boxes, gap_multiplier=1.2):
    if not boxes:
        return []
    
    # Group by approximate lines first
    lines = _group_by_lines(boxes)
    
    groups = []
    for line_words in lines:
        if len(line_words) <= 1:
            groups.extend([[word] for word in line_words])
            continue
        
        # Sort words by x-coordinate
        line_words.sort(key=lambda x: x['bbox'][0])
        
        # Analyze spacing patterns in this line
        line_groups = group_words_in_line(line_words, gap_multiplier)
        groups.extend(line_groups)
    
    return groups

# Merge bounding boxes of words in a group
def merge_group_bboxes(word_group):
    if not word_group:
        return None
    
    # Find the overall bounding box
    x1 = min(box['bbox'][0] for box in word_group)
    y1 = min(box['bbox'][1] for box in word_group)
    x2 = max(box['bbox'][2] for box in word_group)
    y2 = max(box['bbox'][3] for box in word_group)
    
    # Combine text
    combined_text = ' '.join(box['text'] for box in 
                            sorted(word_group, key=lambda x: x['bbox'][0]))
    
    # Average confidence
    avg_conf = sum(box['conf'] for box in word_group) / len(word_group)
    
    return {
        'text': combined_text,
        'bbox': (x1, y1, x2, y2),
        'conf': avg_conf,
        'word_count': len(word_group),
        'inside': False
    }


if __name__ == '__main__':
    app.run(debug=True)