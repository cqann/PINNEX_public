import os
import math
from collections import defaultdict

L0 = 20000
FILE_NAME = "slab"


def cross_product(v1, v2):
    """Calculates the cross product of two 3D vectors."""
    return [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]


def subtract_vectors(v1, v2):
    """Calculates v1 - v2 for 3D vectors."""
    return [v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]]


def dot_product(v1, v2):
    """Calculates the dot product of two 3D vectors."""
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]


def normalize_vector(v):
    """Normalizes a 3D vector (makes its length 1)."""
    if v is None:
        return [0.0, 0.0, 0.0]
    magnitude = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if magnitude == 0:
        return [0.0, 0.0, 0.0]
    return [v[0] / magnitude, v[1] / magnitude, v[2] / magnitude]


def get_data_lines(filepath):
    """
    Generator that yields stripped lines from a file.
    Skips the first line if it's a single integer (assumed to be a row count header).
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return

    with open(filepath, 'r') as f:
        first_line_read = f.readline()
        if not first_line_read:  # Empty file
            return

        first_line_stripped = first_line_read.strip()
        try:
            if len(first_line_stripped.split()) == 1:
                int(first_line_stripped)
                print(f"Info: Skipped potential header line in {filepath}: '{first_line_stripped}'")
            else:
                yield first_line_stripped
        except ValueError:
            yield first_line_stripped

        for line in f:
            yield line.strip()


def read_pts_file(filepath):
    """Reads point coordinates (unscaled) from the .pts file."""
    points = []
    for line_number, line_content_stripped in enumerate(get_data_lines(filepath), start=1):
        if not line_content_stripped:
            continue

        parts = line_content_stripped.split()
        try:
            coords = [float(p) for p in parts]
            if len(coords) == 3:
                points.append(coords)
            else:
                print(
                    f"Info: PTS: Non-point data or malformed line at approx line {line_number}. Assuming end of points. Line: '{line_content_stripped}'")
                break
        except ValueError:
            print(
                f"Info: PTS: Non-coordinate data at approx line {line_number}. Assuming end of points. Line: '{line_content_stripped}'")
            break
    return points


def read_surf_file_for_triangles_and_points(filepath, max_valid_point_index):
    """
    Reads surface triangles from .surf file.
    Returns a list of triangles (each as 3 0-based point indices)
    and a set of unique 0-based point indices involved in valid triangles.
    """
    triangles_as_indices_list = []
    unique_point_indices_set = set()

    for line_number, line_content_stripped in enumerate(get_data_lines(filepath), start=1):
        if not line_content_stripped:
            continue

        if line_content_stripped.startswith("Tr"):
            parts = line_content_stripped.split()
            if len(parts) >= 4:
                try:
                    current_triangle_point_indices = [int(p) for p in parts[1:4]]

                    valid_triangle = True
                    for p_idx in current_triangle_point_indices:
                        if not (0 <= p_idx <= max_valid_point_index):
                            print(f"Warning: SURF: Point index {p_idx} in triangle {parts[1:4]} (from file) "
                                  f"at approx line {line_number} is out of bounds (max valid: {max_valid_point_index}). Skipping this triangle.")
                            valid_triangle = False
                            break

                    if valid_triangle:
                        if len(set(current_triangle_point_indices)) < 3:
                            print(f"Warning: SURF: Degenerate triangle (repeated point indices) {parts[1:4]} "
                                  f"at approx line {line_number}. Skipping this triangle.")
                            valid_triangle = False

                    if valid_triangle:
                        triangles_as_indices_list.append(current_triangle_point_indices)
                        unique_point_indices_set.update(current_triangle_point_indices)

                except (IndexError, ValueError) as e:
                    print(
                        f"Warning: SURF: Could not parse 'Tr' line at approx line {line_number}: '{line_content_stripped}' - {e}")
            else:
                print(f"Warning: SURF: Malformed 'Tr' line at approx line {line_number}: '{line_content_stripped}'")

    return triangles_as_indices_list, unique_point_indices_set


def process_surface_normals(pts_filepath, surf_filepath, output_filepath):
    print(f"Reading points from {pts_filepath}...")
    points_data = read_pts_file(pts_filepath)
    print(f"Found {len(points_data)} points.")
    if not points_data:
        print("Error: No points loaded. Cannot proceed.")
        return

    max_pt_idx = len(points_data) - 1

    print(f"Reading surface triangles from {surf_filepath}...")
    triangles_as_indices, unique_surface_point_indices = read_surf_file_for_triangles_and_points(
        surf_filepath, max_pt_idx)
    print(f"Found {len(triangles_as_indices)} valid triangles and {len(unique_surface_point_indices)} unique surface points.")

    if not triangles_as_indices:
        print("No valid triangles found. Output will be empty.")
        with open(output_filepath, 'w') as f_out:
            pass
        return

    triangle_geo_normals = []
    for i, tri_indices in enumerate(triangles_as_indices):
        p0_idx, p1_idx, p2_idx = tri_indices
        P0, P1, P2 = points_data[p0_idx], points_data[p1_idx], points_data[p2_idx]
        V1 = subtract_vectors(P1, P0)
        V2 = subtract_vectors(P2, P0)
        N_tri = cross_product(V1, V2)
        N_tri_normalized = normalize_vector(N_tri)
        triangle_geo_normals.append(N_tri_normalized)
        if N_tri_normalized == [0.0, 0.0, 0.0]:
            print(
                f"Info: Triangle {i} (indices {tri_indices}) resulted in a zero normal (likely degenerate geometry / zero area).")

    point_to_incident_triangle_normals = defaultdict(list)
    for i, tri_indices in enumerate(triangles_as_indices):
        normal_of_this_triangle = triangle_geo_normals[i]
        for point_idx in tri_indices:
            if normal_of_this_triangle != [0.0, 0.0, 0.0]:
                point_to_incident_triangle_normals[point_idx].append(normal_of_this_triangle)

    final_point_normals = {}
    for point_idx in unique_surface_point_indices:
        original_incident_normals = point_to_incident_triangle_normals.get(point_idx, [])

        if original_incident_normals:
            # Determine reference normal for re-orientation
            ref_normal_sum = [sum(coords) for coords in zip(*original_incident_normals)]
            ref_normal = normalize_vector(ref_normal_sum)

            if ref_normal == [0.0, 0.0, 0.0]:  # Fallback if sum cancelled out
                ref_normal = original_incident_normals[0]  # First normal is guaranteed non-zero here

            reoriented_normals = []
            for n_vec in original_incident_normals:
                if dot_product(n_vec, ref_normal) < 0:
                    reoriented_normals.append([-n_vec[0], -n_vec[1], -n_vec[2]])
                else:
                    reoriented_normals.append(n_vec)

            # Average the reoriented normals
            avg_sum_x, avg_sum_y, avg_sum_z = 0.0, 0.0, 0.0
            for n_vec in reoriented_normals:
                avg_sum_x += n_vec[0]
                avg_sum_y += n_vec[1]
                avg_sum_z += n_vec[2]

            num_incident = len(reoriented_normals)
            avg_normal = [avg_sum_x / num_incident, avg_sum_y / num_incident, avg_sum_z / num_incident]
            final_point_normals[point_idx] = normalize_vector(avg_normal)
        else:
            final_point_normals[point_idx] = [0.0, 0.0, 0.0]

    # MODIFICATION: Filter points for output based on new threshold
    eligible_points_for_output = []
    for point_idx in unique_surface_point_indices:
        num_incident_triangles = len(point_to_incident_triangle_normals.get(point_idx, []))
        if num_incident_triangles > 5:  # CHANGED THRESHOLD
            if point_idx in final_point_normals and 0 <= point_idx < len(points_data):
                eligible_points_for_output.append(point_idx)
            # else: # This case should be rare if point_idx came from unique_surface_point_indices
            #     print(f"Internal Info: Point {point_idx} met >6 triangle criteria but missing from final_point_normals or points_data. Skipping from output.")

    print(
        f"Filtered down to {len(eligible_points_for_output)} points (belonging to >6 non-degenerate triangles) for output.")

    if not eligible_points_for_output:
        print("No points met the criteria of belonging to more than 6 triangles. Output file will be empty.")
        with open(output_filepath, 'w') as f_out:
            pass
        return
    scaled_points_for_output = [[c / L0 for c in p] for p in points_data]

    print(f"Writing output to {output_filepath}...")
    with open(output_filepath, 'w') as f_out:
        for point_idx in sorted(eligible_points_for_output):
            scaled_coord = scaled_points_for_output[point_idx]
            normal_coord = final_point_normals.get(point_idx, [0.0, 0.0, 0.0])

            f_out.write(f"{scaled_coord[0]:.6f} {scaled_coord[1]:.6f} {scaled_coord[2]:.6f}  "
                        f"{normal_coord[0]:.6f} {normal_coord[1]:.6f} {normal_coord[2]:.6f}\n")

    print(f"Processing complete. Output written to {output_filepath}")


if __name__ == "__main__":
    pts_file = FILE_NAME + ".pts"
    surf_file = FILE_NAME + ".surf"
    output_file = FILE_NAME + "_coll_pts_normals.txt"  # Updated filename

    required_files = [pts_file, surf_file]
    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("Error: The following input files were not found:")
        for f_path in missing_files:
            print(f"- {os.path.abspath(f_path)}")
        print("Please ensure these files exist in the correct location or update the file paths in the script.")
    else:
        process_surface_normals(pts_file, surf_file, output_file)
