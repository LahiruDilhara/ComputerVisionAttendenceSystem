import cv2
import numpy as np
import argparse
from rich_argparse import RichHelpFormatter
from pathlib import Path

def parser():
    parser = argparse.ArgumentParser(
        description="Student Attendance Management System",
        formatter_class=RichHelpFormatter,
        usage="python3 sams.py [-h] <image_file> <xml_file>"
    )

    parser.add_argument("image",type=Path,help="Path to the image file")
    parser.add_argument("xml",type=Path,help="Path to the student xml file")

    args = parser.parse_args()
    return args.image, args.xml

def validate(imagePath: Path, xmlPath: Path):
    # Check if the image file exists
    if not imagePath.exists():
        print("Image file does not exist")
        return False
    # Check if the xml file exists
    if not xmlPath.exists():
        print("Xml file does not exist")
        return False
    # Check whether image redable
    try:
        img = cv2.imread(str(imagePath))
        if img is None:
            print("Image file is not readable")
            return False
    except Exception as e:
        print(e)
        return False
    return True

def processAttendance():

    pass

def main():
    
    (imagePath, xmlPath) = parser()
    if not validate(imagePath,xmlPath):
        return
    processAttendance()



if __name__ == "__main__":
    main()