import argparse
import torch
from PIL import Image, ImageDraw, ImageFont
from pdf2image import convert_from_path
import os
import csv

# Load the model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best_4_label.pt')

def calculate_iou(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)

    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area if union_area != 0 else 0

    return iou

def detect_and_save(image_path, page_number, pdf_filename, save_dir, save_image):
    font = ImageFont.truetype("/usr/share/fonts/truetype/fonts-yrsa-rasa/Rasa-Regular.ttf", 30)
    
    image = Image.open(image_path)
    results = model(image)
    df = results.pandas().xyxy[0]

    output_results = {
        'sign': "NG",
        'duty_stamp': "NG",
        'ds_number': "NG",
        'e_duty_stamp': "NG",
        'IoU_sign_dstamp': "NG",
        'IoU_sign_e_dstamp': "NG",
        'Validation Status': "Not Accepted",
    }

    detected_labels = df['name'].tolist()
    for label in output_results.keys():
        if label in detected_labels:
            output_results[label] = "OK"

    bboxes = {row['name']: [row['xmin'], row['ymin'], row['xmax'], row['ymax']] for _, row in df.iterrows()}
    iou_sign_duty = calculate_iou(bboxes.get('sign', [0, 0, 0, 0]), bboxes.get('duty_stamp', [0, 0, 0, 0]))
    iou_sign_e_duty = calculate_iou(bboxes.get('sign', [0, 0, 0, 0]), bboxes.get('e_duty_stamp', [0, 0, 0, 0]))

    iou_threshold = 0.10
    if iou_sign_duty > iou_threshold:
        output_results['IoU_sign_dstamp'] = "OK"

    if iou_sign_e_duty <= 0 and sum(bboxes.get('e_duty_stamp', [0, 0, 0, 0])) != 0:
        output_results['IoU_sign_e_dstamp'] = "OK"

    if output_results['IoU_sign_dstamp'] == "OK" and output_results['sign'] == "OK" and output_results['duty_stamp'] == "OK":
        output_results['Validation Status'] = "Accepted"
        
    if output_results['IoU_sign_e_dstamp'] == "OK" and output_results['sign'] == "OK" and output_results['e_duty_stamp'] == "OK":
        output_results['Validation Status'] = "Accepted"

    save_type = "Accepted" if output_results['Validation Status'] == "Accepted" else "Refused"
    
    if save_image:
        draw = ImageDraw.Draw(image)
    
        for _, row in df.iterrows():
            xmin, ymin, xmax, ymax = row['xmin'], row['ymin'], row['xmax'], row['ymax']
            label = row['name']

            if label == "sign":
                color = "blue"
            elif label == "duty_stamp":
                color = "orange"
            elif label == "e_duty_stamp":
                color = "red"
            elif label == "ds_number":
                color = "green"
            else:
                color = "white"

            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

            text = label
            text_bbox = draw.textbbox((xmin, ymin - 20), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_background = [xmin, ymin - text_height - 10, xmin + text_width, ymin]
            draw.rectangle(text_background, fill=color)

            draw.text((xmin + 2, ymin - text_height - 8), text, font=font, fill="white")
        
        # AI rendering text position
        text_x = 100
        text_y = 10
        padding = 20

        for label, status in output_results.items():
            text = f"{label}: {status}"

            if label == "IoU_sign_dstamp" or label == "IoU_sign_e_dstamp":
                text_color = 'white' if status == "OK" else 'black'
                bg_color = 'blue' if status == "OK" else 'lightcoral'

            elif label == "Validation Status":
                text_color = 'black' if status == "Accepted" else 'black'
                bg_color = 'green' if status == "Accepted" else 'lightcoral'
                
            else:
                text_color = 'black' if status == "OK" else 'black'
                bg_color = 'lightgreen' if status == "OK" else 'lightcoral'

            bbox = draw.textbbox((text_x, text_y), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            bg_width = text_width + 2 * padding
            bg_height = text_height + 2 * padding

            draw.rectangle([text_x, text_y, text_x + bg_width, text_y + bg_height], fill=bg_color)

            draw.text((text_x + padding, text_y + padding), text, font=font, fill=text_color)
            text_y += bg_height  # Move to next line

        image_filename = f"{os.path.splitext(pdf_filename)[0]}_page_{page_number}.png"
        saved_image_path = os.path.join(save_dir, image_filename)
        image.save(saved_image_path)

    return output_results['Validation Status']

def process_pdfs_in_folder(folder_path, save_dir, csv_file, save_image):
    if not os.path.exists(save_dir) and save_image:
        os.makedirs(save_dir)

    all_results = []

    for pdf_file in os.listdir(folder_path):
        if pdf_file.endswith('.pdf'):
            pdf_path = os.path.join(folder_path, pdf_file)
            images = convert_from_path(pdf_path)

            pdf_status = "Refused"
            for i, image in enumerate(images):
                image.save("temp_page.png")
                page_status = detect_and_save("temp_page.png", i + 1, pdf_file, save_dir, save_image)
                
                if page_status == "Accepted":
                    pdf_status = "Accepted"
                
                os.remove("temp_page.png")

            all_results.append([pdf_file, pdf_status])
            print(pdf_file,"_",pdf_status)

    # Write all results to the CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['PDF File', 'Validation Status'])
        writer.writerows(all_results)

    print(f"All results saved to {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process PDF documents and optionally save images.")
    parser.add_argument('path_pdf_documents', type=str, help='Path to the folder containing PDF documents')
    parser.add_argument('--save_image', type=bool, default=False, help='Whether to save images to ai_rendering folder')

    args = parser.parse_args()

    folder_path = args.path_pdf_documents
    save_dir = './ai_rendering'   
    csv_file = './report/final_results.csv' 

    process_pdfs_in_folder(folder_path, save_dir, csv_file, args.save_image)