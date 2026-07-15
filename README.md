# Computer Vision Attendance System

This project extracts attendance from a scanned signing sheet image and a companion XML file, then stores the results in a local SQLite database. It is designed for sheets that contain a table of student signatures. The current implementation analyzes the right-most signature column in the image, detects which signature boxes are filled, and maps those detections to the students listed in the XML file.

## How it works

The main workflow is implemented in `sams.py`:

1. Read an attendance sheet image and an XML file from the command line.
2. Validate the image name, file format, and XML structure.
3. Parse the XML to get the batch name and the ordered student list.
4. Process the image with OpenCV to detect the signature boxes.
5. Decide which boxes are signed by comparing pixel density inside each box.
6. Write the result into `./db/attendance.db` as `Present` or `Absent`.

If `--show_image` is enabled, the script also writes a processed preview image to `output.png` with the detected signature boxes highlighted.

## Project Structure

- `sams.py` - command-line entry point and image processing pipeline.
- `repository.py` - SQLite database layer for batches, students, and attendance records.
- `main.py` - placeholder script that currently only prints a greeting.
- `xml_schema.xsd` - XML schema used to validate the input XML file.
- `public/img/` - sample signing-sheet images.
- `public/xml/info.xml` - sample XML file with the student list.
- `db/` - created automatically for the SQLite database file.

## Requirements

- Python 3.13 or newer
- OpenCV
- NumPy
- Rich
- rich-argparse
- xmlschema

The project already includes a `pyproject.toml` and `uv.lock`, so `uv` is the easiest way to set up and run it.

## Install Dependencies

From the project root:

```bash
uv sync
```

If you prefer to run inside an activated virtual environment, install dependencies with your usual Python workflow instead.

## Run the Project

The script expects two positional arguments:

```bash
python sams.py <image_file> <xml_file>
```

The image file name must follow this pattern:

```text
dd.mm.yyyy.bXX.<extension>
```

Examples:

```bash
python sams.py public/img/10.07.2019.b15.jpeg public/xml/info.xml
python sams.py public/img/11.07.2019.b15.jpeg public/xml/info.xml --show_image
```

Using `uv`:

```bash
uv run python sams.py public/img/10.07.2019.b15.jpeg public/xml/info.xml
uv run python sams.py public/img/10.07.2019.b15.jpeg public/xml/info.xml --show_image
```

When `--show_image` is used, the script saves a preview as `output.png` in the current directory.

## Input Format

### Image

The image should be a scanned or photographed attendance sheet that contains a table with student signature boxes. The script focuses on the right-most column of signature boxes and expects the number of detected boxes to match the number of students in the XML file.

### XML

The XML file must match the schema in `xml_schema.xsd`. The sample file in `public/xml/info.xml` shows the expected structure:

```xml
<nsbm>
	<batches>
		<b15>
			<students>
				<student>
					<index>10000409</index>
					<name>Dilshanika Perera</name>
				</student>
			</students>
		</b15>
	</batches>
</nsbm>
```

The batch tag name must match the batch code in the image file name. For example, `10.07.2019.b15.jpeg` maps to batch `b15`.

## Output

After a successful run, attendance is stored in `./db/attendance.db` with these tables:

- `batch`
- `student`
- `attendance`

Each student is marked `Present` if their signature box is detected as filled, otherwise `Absent`.

## Sample Data

The repository includes sample sheet images in `public/img/` and a sample XML file in `public/xml/info.xml`. A good starting command is:

```bash
uv run python sams.py public/img/10.07.2019.b15.jpeg public/xml/info.xml --show_image
```

## Notes

- The script will stop if the image file, XML file, or batch name does not match the required format.
- If the number of detected signature boxes does not match the number of students in the XML, the run is rejected.
- The database file is created automatically the first time you run the script.
