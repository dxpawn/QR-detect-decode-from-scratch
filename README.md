# AIT3002 Midterm Assignment: QR Code Detection and Decoding

## Problem Description
In this project, you are required to build an image processing program capable of:
1. **Counting** the number of QR codes appearing in each image.
2. **Locating** each QR code (coordinates of the 4 corners of the bounding box).
3. **Extracting** the content of each QR code (optional/bonus).

Each image in the dataset may contain zero, one, or multiple QR codes. These codes appear in various sizes, positions, and lighting conditions.

## Scoring Criteria
* **Report:** 40%
* **Code Quality:** 20%
* **Ranking:** 20% (Based on speed and accuracy/F1-score on a private test set).
* **Implementation from Scratch:** 20% (No specialized QR code libraries allowed).
* **Bonus (+20%):** If the decoding logic is also implemented from scratch.
    * *Note:* Decoding will not be included in the speed ranking. It must be toggleable via an option flag: `--decode=no|yes`.

## Dataset

[Download the dataset here](https://drive.google.com/file/d/198R9LtEwS3wnoadc0gFxi4bSKBDc13VH/view)

The data is divided into three parts:
* **Public Train:** $N_1$ images.
* **Public Valid:** $N_2$ images.
* **Private Test:** $N_3$ images.

Each image corresponds to a row in the input CSV file.

## Requirements
For every image provided, the program must:
1. Detect the total number of QR codes.
2. Determine the precise location (4 corner coordinates) of each QR code.
3. Decode the content of each QR code (optional).

## Execution Instructions
The program must be executable via the command line using the following format:

```bash
python main.py --data public_train.csv
python main.py --data public_valid.csv
```

When evaluating on the private set, the following command will be used:
```bash
python main.py --data private_test.csv
```
The program must function correctly in both cases without any modifications to the source code.

## Output
For each image, the program must output:
* The number of detected QR codes.
* The position (coordinates) of each QR code.
* The decoded content (optional).

The specific format for `output.csv` is detailed in the `output_requirement.md` file (provided in the data folder).

## Evaluation Metrics
The project is evaluated based on two primary pillars:
1. **Accuracy** (F1-score).
2. **Processing Speed**.

The final score will be a weighted combination of these two criteria (specific weights to be announced).

## Submission Requirements
### a) Source Code
* **Main File:** `main.py`
* The script must support the `--data` parameter.
* Ensure the program is not dependent on local environments that are difficult to reproduce.
* Include environment configuration files (e.g., `requirements.txt`, `environment.yaml`, or `pyproject.toml`).

### b) Report
* **Format:** PDF.
* **Content:** Present results and analysis based on the **Public** dataset.
* The **Private** dataset must not be used for tuning or reporting results; it is reserved for independent evaluation.

**Recommended Report Outline:**
1. **Methodology Description:** * Image Pre-processing.
    * QR Code Region Detection.
    * QR Code Decoding.
2. **Technical Stack:** * Libraries/Tools used.
    * Algorithms or models applied.
3. **Public Set Results:** * Accuracy and Processing Speed.
    * Analysis of typical correct/incorrect cases.
4. **Discussion & Analysis:** * Challenges encountered.
    * Methodological limitations.
    * Potential improvements.
```
