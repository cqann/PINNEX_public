import os, sys, csv, math, shutil, subprocess, tempfile, time
import vtk, numpy as np

# ----------------------------------------------------------------------------
#  prepare_openCarp  (verbatim)
# ----------------------------------------------------------------------------
def prepare_openCarp(folder, file_name):
    """
    Original prepare_openCarp code unchanged.
    """
    target_folder  = os.path.abspath(os.path.join(os.path.dirname(__file__), "Meshes", folder))
    output_folder  = os.path.join(target_folder, "openCarp")
    os.makedirs(output_folder, exist_ok=True)

    vtu_path = os.path.join(target_folder, file_name)
    vtk_filename = os.path.join(output_folder, "cobiveco.vtk")

    # Step 1: VTU → VTK
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(vtu_path)
    reader.Update()
    unstructured_grid = reader.GetOutput()

    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(vtk_filename)
    writer.SetInputData(unstructured_grid)
    writer.SetFileTypeToASCII()
    writer.Write()

    # Step 2: meshtool conversion
    input_mesh_basename  = os.path.join(output_folder, "cobiveco")
    output_mesh_basename = os.path.join(output_folder, "heart")
    command = [
        "meshtool", "convert",
        f"-imsh={input_mesh_basename}", "-ifmt=vtk",
        f"-omsh={output_mesh_basename}", "-ofmt=carp_txt",
    ]
    subprocess.run(command, check=True)

    # Step 3: convert .pts to µm
    pts_input      = output_mesh_basename + ".pts"
    pts_output     = os.path.join(output_folder, "heart.pts")
    pts            = np.loadtxt(pts_input, skiprows=1)
    np.savetxt(pts_output, pts*1000.0, header=str(pts.shape[0]), comments='', fmt="%.1f")

    # Step 4 & 5: CSV & cobi.csv (unchanged helper below)
    csv_output = os.path.join(output_folder, "XYZBVC.csv")
    with open(vtk_filename) as f:
        all_lines = f.readlines()
    lines_iter = iter(all_lines)

    # --- identical parse logic ---
    npoints = None
    for line in lines_iter:
        if line.strip().startswith("POINTS"):
            _, n_str, _ = line.strip().split()
            npoints = int(n_str); break
    if npoints is None:
        raise ValueError("No 'POINTS' section found")

    coords = parse_floats(lines_iter, 3*npoints)
    coords = np.round(np.array(coords)*1000, 1)
    x, y, z = coords[0::3], coords[1::3], coords[2::3]

    def skip_until(target):
        for l in lines_iter:
            if l.strip().startswith(target): return
        raise ValueError(f"Field '{target}' not found")

    skip_until("tv "); tv_vals = parse_floats(lines_iter, npoints)
    skip_until("tm "); tm_vals = parse_floats(lines_iter, npoints)
    skip_until("rt "); rt_vals = parse_floats(lines_iter, npoints)
    skip_until("ab");  ab_vals = parse_floats(lines_iter, npoints)

    with open(csv_output, "w") as out:
        out.write("x,y,z,ab,rt,tm,tv\n")
        for i in range(npoints):
            out.write(f"{x[i]},{y[i]},{z[i]},{ab_vals[i]},{rt_vals[i]},{tm_vals[i]},{tv_vals[i]}\n")

    cobi_out = os.path.join(output_folder, "cobi.csv")
    with open(csv_output)             as infile,\
         open(cobi_out, "w", newline="") as outfile:
        reader = csv.DictReader(infile)
        writer = csv.DictWriter(outfile, fieldnames=["ab", "rt", "tm", "tv"])
        writer.writeheader()
        for row in reader:
            writer.writerow({f: row[f] for f in writer.fieldnames})
    print("Pipeline completed successfully.")
    if os.path.exists(vtk_filename):
        os.remove(vtk_filename)
        print(f"Removed temporary file: {vtk_filename}")



# ----------------------------------------------------------------------------
#  convert_vtu_to_vtk · extract_fiber_sheet · run_set_fibers  (verbatim)
# ----------------------------------------------------------------------------
def convert_vtu_to_vtk(input_file: str, output_file: str):
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(input_file)
    reader.Update()
    writer = vtk.vtkUnstructuredGridWriter()
    writer.SetFileName(output_file)
    writer.SetInputData(reader.GetOutput())
    writer.SetFileTypeToASCII()
    writer.Write()

def extract_fiber_sheet(vtk_file: str, output_file: str):
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(vtk_file)
    reader.Update()
    data = reader.GetOutput()

    fiber_array = data.GetCellData().GetArray("Fiber") or data.GetPointData().GetArray("Fiber")
    sheet_array = data.GetCellData().GetArray("Sheet") or data.GetPointData().GetArray("Sheet")
    if fiber_array is None or sheet_array is None:
        print("Fiber/Sheet array not found."); sys.exit(1)

    with open(output_file, "w") as f:
        f.write("2\n")
        for i in range(fiber_array.GetNumberOfTuples()):
            fiber = fiber_array.GetTuple(i)
            sheet = sheet_array.GetTuple(i)
            f.write(f"{fiber[0]:.6f} {fiber[1]:.6f} {fiber[2]:.6f} {sheet[0]:.6f} {sheet[1]:.6f} {sheet[2]:.6f}\n")

def run_set_fibers(folder: str = "Mean"):
    base_dir       = os.path.dirname(__file__)
    target_folder  = os.path.abspath(os.path.join(base_dir, "Meshes", folder))
    input_vtu      = os.path.join(target_folder, "fiber.vtu")
    output_vtk     = os.path.join(target_folder, "heart.vtk")
    output_lon     = os.path.join(target_folder, "openCarp", "heart.lon")
    os.makedirs(os.path.dirname(output_lon), exist_ok=True)
    convert_vtu_to_vtk(input_vtu, output_vtk)
    extract_fiber_sheet(output_vtk, output_lon)
    if os.path.exists(output_vtk):
        os.remove(output_vtk)
        print(f"Removed temporary file: {output_vtk}")


# ----------------------------------------------------------------------------
#  helper used above
# ----------------------------------------------------------------------------
def parse_floats(lines_iter, count):
    values = []
    while len(values) < count:
        values.extend(map(float, next(lines_iter).split()))
    return values[:count]
