
import os, sys, csv, math, shutil, subprocess, tempfile, time
import vtk, numpy as np

def uvc_to_cobi(a: float, r: float, m: float, v: int):
    """Convert UVC → CoBi (Eqs. 51-54)."""
    tv = v
    tm = 1 - m
    ab = a
    if v == 0:
        if abs(r) > math.pi/2:
            rt = 2/3 + 2/(3*math.pi) * math.atan2(math.cos(r), math.sin(r))
        else:
            rt = 2/3 + 1/(3*math.pi) * math.atan2(math.cos(r), math.sin(r))
    elif v == 1:
        rt = 1/3 + 2/(3*math.pi) * r
    else:
        raise ValueError("v must be 0 or 1")
    return ab, rt, tm, tv

def replace_class_label(input_surface, output_surface, from_label=5, to_label=2):
    """
    Reads a VTP surface file, replaces all occurrences of `from_label` with `to_label`
    in the point data array named 'class', and writes the modified surface to output_surface.

    Parameters:
    - input_surface (str): Path to the input VTP file.
    - output_surface (str): Path to save the modified VTP file.
    - from_label (int, optional): The label value to be replaced (default is 5).
    - to_label (int, optional): The new label value (default is 2).
    """
    # Read the surface mesh
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(input_surface)
    reader.Update()
    polydata = reader.GetOutput()

    # Retrieve the 'class' array from point data
    class_array = polydata.GetPointData().GetArray("class")
    if class_array is None:
        raise ValueError(
            "The 'class' array was not found in the point data of the input file."
        )

    # Replace all instances of from_label with to_label
    num_points = polydata.GetNumberOfPoints()
    for i in range(num_points):
        if class_array.GetValue(i) == from_label:
            class_array.SetValue(i, to_label)

    # Mark the array as modified
    class_array.Modified()

    # Write out the modified surface mesh
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(output_surface)
    writer.SetInputData(polydata)
    writer.Write()


def run_cobiveco(
    volume_file,
    new_surface_file,
    matlab_folder,
):
    """
    Copies the provided volume and surface files into the correct subfolder expected by test.m,
    renaming them to 'heart.vtu' and 'heart.vtp', respectively, and then runs the MATLAB script test.m.

    Parameters:
      volume_file (str): Path to the original volume file.
      new_surface_file (str): Path to the modified surface file.
      matlab_folder (str): Path to the MATLAB folder containing test.m.

    Note:
      The MATLAB script test.m expects the input files to be located in:
         matlab_folder/Mean_2.0/heart
      and to be named 'heart.vtu' (for the volume file) and 'heart.vtp' (for the surface file).
    """
    if not os.path.isdir(matlab_folder):
        raise ValueError(f"MATLAB folder '{matlab_folder}' does not exist.")

    input_folder = os.path.join(matlab_folder, "Mesh")
    os.makedirs(input_folder, exist_ok=True)

    # Define destination file paths in the input folder
    volume_dest = os.path.join(input_folder, "heart.vtu")
    surface_dest = os.path.join(input_folder, "heart.vtp")

    shutil.copy(volume_file, volume_dest)
    shutil.copy(new_surface_file, surface_dest)

    print(f"Copied volume file from '{volume_file}' to '{volume_dest}'")
    print(f"Copied surface file from '{new_surface_file}' to '{surface_dest}'")


    result_file = os.path.join(matlab_folder, "Result", "result.vtu")
    dest_folder = os.path.dirname(volume_file)
    dest_result_file = os.path.join(dest_folder, "cobiveco.vtu")
    try:
        subprocess.run(["matlab", "-batch", "cobi"], cwd=matlab_folder, check=True)
    except subprocess.CalledProcessError as e:
        print("MATLAB execution failed.")
        raise e

    shutil.copy(result_file, dest_result_file)
    print(f"Copied result file from '{result_file}' to '{dest_result_file}'")


def run_fibers(
    volume_file,
    surface_file,
    matlab_folder,
):
    """
    Copies the volume and surface files into the correct subfolder for the MATLAB fiber processing,
    runs the MATLAB script 'assign_fiber.m', and then copies the result file to the volume file's folder.

    Parameters:
      volume_file (str): Path to the original volume file.
      surface_file (str): Path to the original surface file (expected to be named 'heart_sur.vtp' or similar).
      matlab_folder (str): Path to the MATLAB folder containing 'assign_fiber.m'.

    Process:
      1. Copy volume_file into matlab_folder/Mesh as 'heart.vtu'.
      2. Copy surface_file into matlab_folder/Mesh as 'heart_sur.vtp'.
      3. Run MATLAB in batch mode to execute the 'assign_fiber' script.
      4. Copy the result file matlab_folder/Result/heart.vtu to the directory of volume_file and rename it to 'fiber.vtu'.
    """
    # Verify that the MATLAB folder exists.
    if not os.path.isdir(matlab_folder):
        raise ValueError(f"MATLAB folder '{matlab_folder}' does not exist.")

    # Define the input folder (as expected by assign_fiber.m).
    input_folder = os.path.join(matlab_folder, "Mesh")
    os.makedirs(input_folder, exist_ok=True)

    # Define destination file paths in the input folder.
    volume_dest = os.path.join(input_folder, "heart.vtu")
    surface_dest = os.path.join(input_folder, "heart.vtp")

    # Copy the volume file as 'heart.vtu' into the Mesh folder.
    shutil.copy(volume_file, volume_dest)
    # Copy the surface file as 'heart_sur.vtp' into the Mesh folder.
    shutil.copy(surface_file, surface_dest)

    print(f"Copied volume file from '{volume_file}' to '{volume_dest}'")
    print(f"Copied surface file from '{surface_file}' to '{surface_dest}'")

    # Run MATLAB in batch mode to execute 'assign_fiber.m'.
    # The working directory is set to matlab_folder.
    try:
        subprocess.run(
            ["matlab", "-batch", "assign_fiber"], cwd=matlab_folder, check=True
        )
    except subprocess.CalledProcessError as e:
        print("MATLAB execution failed.")
        raise e

    # After MATLAB execution, copy the result file from the Result folder.
    # The MATLAB script produces 'heart.vtu' in matlab_folder/Result.
    result_file = os.path.join(matlab_folder, "Result", "heart.vtu")
    if not os.path.exists(result_file):
        raise FileNotFoundError(
            f"Result file '{result_file}' not found. Check MATLAB output for errors."
        )

    # Determine the destination folder: same as the directory of the volume_file.
    dest_folder = os.path.dirname(volume_file)
    dest_result_file = os.path.join(dest_folder, "fiber.vtu")

    shutil.copy(result_file, dest_result_file)
    print(f"Copied result file from '{result_file}' to '{dest_result_file}'")



def prepare_openCarp(folder, file_name):
    """
    Runs the openCarp preparation pipeline.
    
    Parameters:
      folder (str): The name of the folder (e.g. "Mean") where the input VTU file is located.
      file_name (str): The name of the input VTU file (default is "cobiveco.vtu").
    
    The script:
      - Creates an "openCarp" subfolder inside the target folder.
      - Converts the VTU file to a legacy VTK file.
      - Uses meshtool to convert the VTK file to OpenCarp text format.
      - Converts the .pts file from meshtool output to micrometer units.
      - Generates CSV files from the VTK file.
      - Creates a cobi CSV file with a subset of columns.
    """
    # Construct the target folder (where the input file is located)
    target_folder = os.path.abspath(os.path.join(os.path.dirname(__file__),"Meshes",folder))
    output_folder = os.path.join(target_folder, "openCarp")
    os.makedirs(output_folder, exist_ok=True)

    vtu_path = os.path.join(target_folder, file_name)

    vtk_filename = os.path.join(output_folder, "cobiveco.vtk")
    print("Step 1: Converting VTU to VTK...")

    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()  
    unstructured_grid = reader.GetOutput()

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(vtk_filename)
    writer.SetInputData(unstructured_grid)
    writer.SetFileTypeToASCII()
    writer.Write()  
    print(f"VTK file created: {vtk_filename}")
    
    # --- Step 2: VTK to OpenCarp text conversion via meshtool ---
    input_mesh_basename = os.path.join(output_folder, "cobiveco")
    output_mesh_basename = os.path.join(output_folder, "heart")
    print("Step 2: Converting VTK to OpenCarp text using meshtool...")
    # Build the meshtool command
    command = [
        "meshtool",
        "convert",
        f"-imsh={input_mesh_basename}",
        f"-ifmt=vtk",
        f"-omsh={output_mesh_basename}",
        f"-ofmt=carp_txt",  # Use carp_txt as the output format for OpenCarp text
    ]
    print("Executing command:", " ".join(command))

    try:
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        print("Meshtool conversion successful.")
        cleaned_stdout = "\n".join(
            line for line in result.stdout.splitlines() if line.strip()
        )
        print(cleaned_stdout)
    except subprocess.CalledProcessError as e:
        print("Meshtool conversion failed with the following error:")
        print(e.stderr)
        raise
    # --- Step 3: Convert .pts file from meshtool to micrometer units ---
    pts_input = output_mesh_basename + ".pts"
    output_file = os.path.join(output_folder, "heart.pts")
    print("Step 3: Converting .pts file to micrometer units...")
    # Load the .pts file, skipping the header.
    pts = np.loadtxt(pts_input, skiprows=1)
    conversion_factor=1000.0

    # Apply the conversion (e.g., mm to microns)
    pts_converted = pts * conversion_factor

    # Write the new file: first line is the number of points,
    # followed by the converted points.
    with open(output_file, "w") as f:
        f.write(f"{pts_converted.shape[0]}\n")
        np.savetxt(f, pts_converted, fmt="%.1f")
    

    # --- Step 4: Create CSV file from VTK file (full data) ---
    csv_output = os.path.join(output_folder, "XYZBVC.csv")
    print("Step 4: Converting VTK to CSV...")
    
    with open(vtk_filename, "r") as f:
        all_lines = f.readlines()
    lines_iter = iter(all_lines)

    # 1) Skip lines until we find "POINTS"
    npoints = None
    for line in lines_iter:
        line_stripped = line.strip()
        if line_stripped.startswith("POINTS"):
            # e.g. "POINTS 478820 float"
            _, n_str, dtype = line_stripped.split()
            npoints = int(n_str)
            break

    if npoints is None:
        raise ValueError("No 'POINTS' section found in the file.")

    # 2) Read the coordinates: 3 * npoints floats
    coords = parse_floats(lines_iter, 3 * npoints)
    coords = np.round(np.array(coords) * 1000, 1)
    x_coords = coords[0::3]
    y_coords = coords[1::3]
    z_coords = coords[2::3]

    # Helper function to skip forward until we see a line starting with a target
    def skip_until_field(target):
        for l in lines_iter:
            if l.strip().startswith(target):
                return
        raise ValueError(f"Field '{target}' not found in expected location.")

    # 3) Read each of the 5 pre-POINT_DATA arrays:
    #    ab, rtSin, rtCos, rt, tm
    skip_until_field("tv ")
    tv_vals = parse_floats(lines_iter, npoints)

    skip_until_field("tm ")
    tm_vals = parse_floats(lines_iter, npoints)

    skip_until_field("rt ")
    rt_vals = parse_floats(lines_iter, npoints)

    skip_until_field("ab")
    ab_vals = parse_floats(lines_iter, npoints)


    # 5) Write out to CSV
    with open(csv_output, "w") as out:
        # header
        out.write("x,y,z,ab,rt,tm,tv\n")
        # data rows
        for i in range(npoints):
            out.write(
                f"{x_coords[i]},{y_coords[i]},{z_coords[i]},"
                f"{ab_vals[i]},{rt_vals[i]},"
                f"{tm_vals[i]},{tv_vals[i]}\n"
            )
     # --- Step 5: Create cobi CSV file (selected columns) ---
    print("Step 5: Creating cobi file...")
    # Determine the directory where the input file is located
    directory = os.path.dirname(os.path.abspath(csv_output))
    # Define the output file path (same directory, file named 'cobi.csv')
    output_file = os.path.join(directory, "cobi.csv")

    # Open the input file for reading
    with open(csv_output, "r", newline="") as infile:
        reader = csv.DictReader(infile)
        # Define the fields (columns) to keep
        fields_to_keep = ["ab", "rt", "tm", "tv"]

        # Open the output file for writing
        with open(output_file, "w", newline="") as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fields_to_keep)
            writer.writeheader()  # Write the header row

            # For each row in the input, extract only the desired fields
            for row in reader:
                new_row = {field: row[field] for field in fields_to_keep}
                writer.writerow(new_row)

    print(f"Output file created: {output_file}")

    print("Pipeline completed successfully.")


def convert_vtu_to_vtk(input_file: str, output_file: str) -> None:
    """
    Converts a VTU file to a VTK file in ASCII format.
    """
    # Read the VTU file using VTK
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(input_file)
    reader.Update()  # Read the file

    # Write the file in VTK ASCII format
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(reader.GetOutput())
    writer.SetFileTypeToASCII()  # Write in ASCII format
    writer.Write()

    print(f"Converted '{input_file}' to ASCII VTK file '{output_file}'.")

def extract_fiber_sheet(vtk_file: str, output_file: str) -> None:
    """
    Reads a VTK file, extracts the 'Fiber' and 'Sheet' arrays,
    and writes them in the openCARP .lon format.
    
    The .lon file has a header (2) followed by one line per entry:
      f0_x f0_y f0_z s0_x s0_y s0_z
    """
    # Read the VTK file (legacy format)
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_file)
    reader.Update()
    data = reader.GetOutput()

    # Try to get the arrays from CELL_DATA; if not found, try POINT_DATA
    fiber_array = data.GetCellData().GetArray("Fiber")
    if fiber_array is None:
        fiber_array = data.GetPointData().GetArray("Fiber")
    if fiber_array is None:
        print("Error: 'Fiber' array not found in cell or point data.")
        sys.exit(1)

    sheet_array = data.GetCellData().GetArray("Sheet")
    if sheet_array is None:
        sheet_array = data.GetPointData().GetArray("Sheet")
    if sheet_array is None:
        print("Error: 'Sheet' array not found in cell or point data.")
        sys.exit(1)

    num_entries = fiber_array.GetNumberOfTuples()
    print(f"Found {num_entries} fiber entries.")

    # Write to the .lon file following openCARP format
    with open(output_file, "w") as f:
        f.write("2\n")
        for i in range(num_entries):
            fiber = fiber_array.GetTuple(i)
            sheet = sheet_array.GetTuple(i)
            # Format the six components (3 for fiber, 3 for sheet)
            line = f"{fiber[0]:.6f} {fiber[1]:.6f} {fiber[2]:.6f} " \
                   f"{sheet[0]:.6f} {sheet[1]:.6f} {sheet[2]:.6f}\n"
            f.write(line)

    print(f"Fiber and sheet data successfully written to '{output_file}'.")

def run_set_fibers(folder: str = "Mean") -> None:
    """
    Executes the fiber processing pipeline.
    
    This pipeline:
      1) Converts the VTU file (fiber.vtu) to an ASCII VTK file.
      2) Extracts the 'Fiber' and 'Sheet' arrays from the VTK file and writes them
         to an openCARP .lon file.
    
    The file paths are based on a folder parameter (e.g. "Mean") located under the Meshes directory.
    """
    # Construct the target folder (where the input file is located)
    base_dir = os.path.dirname(__file__)
    target_folder = os.path.abspath(os.path.join(base_dir, "Meshes",folder))
    
    # Define file paths for input and outputs
    input_vtu = os.path.join(target_folder, "fiber.vtu")
    output_vtk = os.path.join(target_folder, "heart.vtk")
    output_lon = os.path.join(target_folder, "openCarp", "heart.lon")
    
    # Ensure the output directory for the .lon file exists
    os.makedirs(os.path.dirname(output_lon), exist_ok=True)
    
    # Step 1: Convert the VTU file to an ASCII VTK file
    convert_vtu_to_vtk(input_vtu, output_vtk)
    
    # Step 2: Extract the fiber and sheet arrays and write to a .lon file
    extract_fiber_sheet(output_vtk, output_lon)


def parse_floats(lines_iter, count):
    """
    Read 'count' floating-point numbers total from the lines iterator.
    Each line may contain multiple floats separated by whitespace.
    """
    values = []
    while len(values) < count:
        line = next(lines_iter).strip()
        parts = line.split()
        for p in parts:
            values.append(float(p))
    return values[:count]