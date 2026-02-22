# Digital Cross Section Mapping Workflow

This repository contains the source code (source.py) for the custom Python program described in:

**Kutaini M, Reich S. Digital Cross Section Mapping: A Novel 3D Slicing and Registration Workflow for Precise Geometrical Analysis in Digital Dentistry. 2026.**

The rest of the files are a more organised, newer, version with a better UI.

## Demo
<img width="1134" height="870" alt="image" src="https://github.com/user-attachments/assets/e696b941-6076-4041-9e5d-285c77832fbf" />

## Requirements
You need Three STL files per abutment:
1. intraoral digital impression of the abutment,
2. digital impression of the same abutment obtained from the conventional master cast, and
3. intraoral digital impression including surrounding structures.

## Usage

### Step 1
First, make sure you have Python installed. This project was successfully tested with Python 3.11.

### Step 2 
Modify the following paths to match your file structure:
(1) in main.py:
Put the three following STL files in one folder, ex.: “model_1”:
- model_file: Path to the intraoral digital impression including surrounding structures. (STL)
- prep_file: Path to the intraoral digital impression (STL)
- prep_k_file: Path to the conventional scanned preparation (STL)

(2) in utils_mesh.py
- icp_filename: Keep the “/icp_points_{tooth_id}.npy” suffix, just change “Zähne/{v}/{xx}/“ to match the path of the three STL files, ex.: “model_1” again.

(3) in gui.py
- icp_filename: Same as above

### Step 3
Once the paths are in place, run "main.py", it will open prep_file and prep_k_file in a new window. 
1. On the blue model (prep_k_file): Left click three distinctable structures
2. On the red model (prep_file): Right click the same three distinctable structures
3. Press Enter

This is the ICP point determination (to find the overlapping points between both files) and is done once. Once you press enter, a .npy file is saved (Path=icp_filename) and is loaded when you run main.py for the same tooth again.

A new window (See: Demo) automatically opens. 

<img width="405" height="339" alt="image" src="https://github.com/user-attachments/assets/fa70e1b8-1fbe-49ec-a967-f23d1cd2910a" />

### Step 4
Rotate the rotation angle (Z) slider at the bottom to view different angles.

### Optional
- If you would like to save the data as .csv, click "Sava Data". This will generate a CSV file with all coordinates and distances in px and mm in a 5° interval automatically.
- If you would like to view the current placement of the 2D slice, click on "3D Preview".

## License
This software is released under a **Custom Non-Commercial License**.  
You may use and modify it for research and educational purposes with proper citation.  
Commercial use requires prior written consent from the author.

The full license is available in the LICENSE.md file in the root directory of this source tree.

© 2026 Majed Kutaini, RWTH Aachen University  
Contact: kutainimajed@gmail.de
