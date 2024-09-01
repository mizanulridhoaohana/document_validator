# Autonomus Document Validator

This project is designed to automate the process for document administrators by verifying the authenticity of documents. Specifically, a document is considered valid if it contains both a stamp and a signature.

To achieve this, I used the advanced YOLOv5 architecture. This choice was driven by YOLOv5's capability to perform both object detection and classification simultaneously with high accuracy. The system efficiently identifies and verifies essential elements on documents, ensuring that each processed document meets the required authenticity standards.

## Project Structure
- `app.py`: The main script for processing PDF documents.
- `models/best.pt`: The trained YOLOv5 model (ensure this file is available in the `models` folder).
- `ai_rendering`: Folder to save rendered image output.
- `pdf_documents`: Folder containing the set of pdfs to be checked.
- `report`: Folder containing the report results.

## How to Run
The script can be run from the terminal with several options. Here’s how to use it:

### Without Saving Images
To process PDF documents without saving the detection images, run the following command:
```python3 app.py path_pdf_documents```
Replace `path_pdf_documents` with the path to the folder containing the PDF files you want to process.

### Saving Images
To process PDF documents and save detection images to the `ai_rendering` folder, run the following command:
```python3 app.py path_pdf_documents --save_image=True```
Replace `path_pdf_documents` with the path to the folder containing the PDF files you want to process.

## Output
- **`ai_rendering` Folder**: This folder will contain the detection images if the `--save_image=True` option is used.
- **`final_results.csv`**: This CSV file contains the validation results for each PDF document.

## Example
For example, if the PDF documents are located in the `pdf_documents` folder and you want to save the detection images, run:
```python3 app.py pdf_documents --save_image=True```

If you do not want to save the images, simply run:
```python3 app.py pdf_documents```

## Notes
- Ensure that the path to the YOLOv5 model (`models/best.pt`) and the PDF folder are correct before running the script.
- Detection results are saved as PNG files in the `ai_rendering` folder if the `--save_image=True` option is used.
