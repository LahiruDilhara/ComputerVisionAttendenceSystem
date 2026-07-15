import cv2
import numpy as np
import argparse
from rich import print
from rich_argparse import RichHelpFormatter
from pathlib import Path
import xml.etree.ElementTree as ET
import xmlschema
import re
import sys
from repository import AttendanceRepository

class CliArgumentParser():
    def __init__(self):
        print("Parsing cli arguments")
        (imagePath, xmlPath, showImage) = self.cliArgumentParser()
        self.imagePath = imagePath
        self.xmlPath = xmlPath
        self.showImage = showImage
        self.date = ""
        self.batch_name = ""
        valid = self.argumentValidator()

        if not valid:
            sys.exit(1)
        self.parsedXML = self.xml_students()
        self.students = self.parseStudentsForBatch()
        

    def cliArgumentParser(self):
        parser = argparse.ArgumentParser(
            description="Student Attendance Management System",
            formatter_class=RichHelpFormatter,
            usage="python3 sams.py [-h] <image_file> <xml_file>"
        )

        parser.add_argument("image",type=Path,help="Path to the image file")
        parser.add_argument("xml",type=Path,help="Path to the student xml file")
        parser.add_argument("-sh","--show_image",action="store_true",help="Show the image file after processing")

        args = parser.parse_args()
        return args.image, args.xml, args.show_image
    
    def argumentValidator(self):
        # Check if the image file exists
        if not self.imagePath.exists():
            print("Image file does not exist")
            return False
            
        # Validate image base file name format
        match = re.fullmatch(r"((\d\d\.){2}\d{4})\.(b\d+)", self.imagePath.stem)
        if not match:
            print("Image base file name does not match the required format")
            return False
        
        self.date = match.group(1)
        self.batch_name = match.group(3)

        # Check if the xml file exists
        if not self.xmlPath.exists():
            print("Xml file does not exist")
            return False
        # Check whether image redable
        try:
            img = cv2.imread(str(self.imagePath))
            if img is None:
                print("Image file is not readable")
                return False
        except Exception as e:
            print("Image file is not in correct format")
            return False
        
        # Check whether xml file format correct
        try:
            tree = ET.parse(str(self.xmlPath))
            root = tree.getroot()
        except Exception as e:
            print("XML file is not redable")
            print(e)
            return False
            
        if not self.xml_format_paser():
            return False
            
        return True
    
    def xml_format_paser(self):
        try:
            schema = xmlschema.XMLSchema("xml_schema.xsd")
            if schema.is_valid(str(self.xmlPath)):
                print("XML file is valid according to the schema.")
                return True
            else:
                print("XML file is invalid according to the schema.")
                schema.validate(str(self.xmlPath))
                return False
        except Exception as e:
            print("XML Layout is invalid")
            return False

    def xml_students(self) -> dict[str, dict[str, list[dict[str, str]]]]:
        tree = ET.parse(str(self.xmlPath))
        root = tree.getroot()
        
        batches_dict: dict[str, list[dict[str, str]]] = {}
        
        batches_node = root.find('batches')
        if batches_node is not None:
            for batch_node in batches_node:
                batch_name = batch_node.tag
                students_list: list[dict[str, str]] = []
                
                students_node = batch_node.find('students')
                if students_node is not None:
                    for student_node in students_node.findall('student'):
                        index_node = student_node.find('index')
                        name_node = student_node.find('name')
                        
                        if index_node is not None and name_node is not None:
                            students_list.append({
                                "index": index_node.text or "",
                                "name": name_node.text or ""
                            })
                
                batches_dict[batch_name] = students_list
        return batches_dict
    
    def parseStudentsForBatch(self):
        if self.batch_name not in self.parsedXML:
            print("The Given Batch not found in xml file")
            sys.exit(1)
        
        if self.parsedXML[self.batch_name] is None:
            print("No students found for the batch")
            sys.exit(1)
        
        return self.parsedXML[self.batch_name]

def processAttendance(imagePath, xmlPath,showImage:bool,attendance_box_count)->list:
    print("Starting the processing")
    image = cv2.imread(str(imagePath),cv2.IMREAD_GRAYSCALE)
    color_image = cv2.imread(str(imagePath))
    print("Read the image data")
    blurred_image = cv2.GaussianBlur(image, (11,11),0)
    print("Applied the gaussian blur to reduce noice")
    binary_image = cv2.adaptiveThreshold(blurred_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,21,2)
    print("Applied adaptive threshold to convert image to black and white")

    print("Creating horizontal kernel to identify horizontal lines")
    hk = cv2.getStructuringElement(cv2.MORPH_RECT,(40,1))
    print("Analyzing horizontal lines")
    hl = cv2.morphologyEx(binary_image,cv2.MORPH_OPEN,hk,iterations=2)

    print("Creating vertical kernel to identify vertical lines")
    vk = cv2.getStructuringElement(cv2.MORPH_RECT,(1,30))
    print("Analyzing vertical lines")
    vl = cv2.morphologyEx(binary_image,cv2.MORPH_OPEN,vk,iterations=2)

    print("Creating table grid")
    table_grid = cv2.addWeighted(hl, 0.5, vl,0.5,0)

    print("Finding the boxes on the image")
    contours, hierarchy = cv2.findContours(table_grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    all_boxes = []
    min_box_height = 50
    min_box_width = 50

    if hierarchy is not None:
        print("Enumerate on found boxes")
        for i,cnt in enumerate(contours):
            print("Identified the deep level boxes")
            if hierarchy[0][i][3] != -1:
                x, y, w, h = cv2.boundingRect(cnt)

                print("Identified the box rotation")
                rect = cv2.minAreaRect(cnt)

                print("Filter the boxes which height or width is larger than threshold")
                if w > min_box_width and h > min_box_height:
                    all_boxes.append((x,y,w,h,rect))
    else:
        print("No boxes were found")
        return []
    
    if not all_boxes or len(all_boxes) == 0:
        print("No boxes were found")
        return []
    
    print("Find the maximum right most edge boxes")
    max_right_edge = max(box[0] + box[2] for box in all_boxes)
    right_edge_threshold = 50

    print("Keep only the rightmost columns")
    right_most_boxes = []
    for box in all_boxes:
        x,y,w,h,rect = box
        if (x + w) > max_right_edge - right_edge_threshold:
            right_most_boxes.append(box)
    
    print(f"Identified right most boxes {len(right_most_boxes)}")
    bottom_boxes = list(reversed(right_most_boxes[0:attendance_box_count]))

    print("Finding the signature identification inner boxes")
    innerPadding = 10
    signature_identification_boxes = []
    for box in bottom_boxes:
        x, y, w, h, rect = box
        (cx, cy), (rw,rh), angle = rect

        new_w = max(1, rw-2*innerPadding)
        new_h = max(1,rh-2*innerPadding)

        inner_rect = ((cx,cy), (new_w,new_h), angle)

        box_points = cv2.boxPoints(inner_rect)
        box_points = np.int32(box_points)
        
        signature_identification_boxes.append(box_points)
    
    print(f"Identified {len(signature_identification_boxes)} signature boxes")
    
    print("Add the pixel density ratio for each box")
    processed_signature_identification_boxes = []
    for points in signature_identification_boxes:
        x, y, w, h = cv2.boundingRect(points)
        cell_roi = binary_image[y:y+h, x:x+w]
        white_pixel_count = cv2.countNonZero(cell_roi)
        black_pixel_count = (cell_roi.shape[0] * cell_roi.shape[1]) - white_pixel_count

        ratio = white_pixel_count / black_pixel_count if black_pixel_count > 0 else 0
        processed_signature_identification_boxes.append({"points":points, "ratio":ratio})
    
    print("Based on the pixel ratio identify the signature signed boxes")
    signature_identified_points = []
    signature_density_threshold = 0.08
    for point in processed_signature_identification_boxes:
        if point["ratio"] > signature_density_threshold:
            signature_identified_points.append({"point":point,"available":True})
        else:
            signature_identified_points.append({"point":point,"available":False})
    
    print("Draw the green boxes on signature identified boxes for visual")
    if (showImage):
        for point in signature_identified_points:
            if point["available"]:
                cv2.drawContours(color_image, [point["point"]["points"]], -1, (0,255,0), 2)
    
        print("Save image")
        output_file = "output.png"
        cv2.imwrite(output_file,color_image)
        print("Image saved")
    return signature_identified_points

def main():
    args = CliArgumentParser()
    repository = AttendanceRepository()
    print()
    attendanceList = processAttendance(args.imagePath,args.xmlPath,args.showImage,attendance_box_count=len(args.students))
    if len(args.students)!= len(attendanceList):
        print("Number of students and signature boxes are not matching")
        sys.exit(1)

    for index,student in enumerate(args.students):
        attendance = True if attendanceList[index]["available"] else False
        student_index = student["index"]
        student_name = student["name"]
        repository.markAttendance(student_index,student_name,args.batch_name,attendance,args.date)
    
if __name__ == "__main__":
    main()