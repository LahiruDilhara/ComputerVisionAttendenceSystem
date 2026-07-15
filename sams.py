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


def main():
    
    (imagePath, xmlPath) = parser()
    print(imagePath)
    print(xmlPath)
    pass



if __name__ == "__main__":
    main()