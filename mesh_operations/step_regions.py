import os, sys, csv, time, tempfile
from math_utils import uvc_to_cobi


# --- helper: .dat mask ------------------------------------------------------
def write_mask_dat_for_csv(csv_path, matching_rows, dat_path="mask.dat"):
    matching_set = set(matching_rows)
    with open(csv_path) as f:
        lines = f.readlines()
    with open(dat_path, "w") as out:
        for i in range(len(lines)):
            out.write(("1" if i in matching_set else "0") + "\n")
    return dat_path


# --- main entry -------------------------------------------------------------
def run_set_regions(folder: str = "Mean", heart="original") -> None:
    target_folder = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "Meshes", heart, folder, "openCarp"
        )
    )
    csv_file = os.path.join(target_folder, "cobi.csv")
    elem_file = os.path.join(target_folder, "heart.elem")
    if not os.path.exists(csv_file):
        print(f"CSV missing: {csv_file}")
        sys.exit(1)
    if not os.path.exists(elem_file):
        print(f"Elem missing: {elem_file}")
        sys.exit(1)

    a_min, a_max = 0.0, 1
    m_min, m_max = 0.0, 1.0
    ab_l, rt_l, tm_l, _ = uvc_to_cobi(a_min, 0, m_min, 0)
    ab_u, rt_u, tm_u, _ = uvc_to_cobi(a_max, 0, m_max, 0)
    ab_bounds = (min(ab_l, ab_u), max(ab_l, ab_u))
    rt_bounds = (0, 1)
    tm_bounds = (min(tm_l, tm_u), max(tm_l, tm_u))
    tv_bounds = (0, 1)

    rows = get_rows_within_coordinates(
        csv_file, ab_bounds, rt_bounds, tm_bounds, tv_bounds
    )
    mask = write_mask_dat_for_csv(
        csv_file, rows, os.path.join(target_folder, "mask.dat")
    )
    update_elem_file_quick(elem_file, rows, region_id=1)
    print("Region masking done.")


# --- helpers kept verbatim --------------------------------------------------
def get_rows_within_coordinates(
    csv_filename, ab_bounds, rt_bounds, tm_bounds, tv_bounds
):
    start = time.time()
    matches = []
    with open(csv_filename, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for idx, row in enumerate(reader):
            try:
                ab = float(row["ab"])
                rt = float(row["rt"])
                tm = float(row["tm"])
                tv = float(row["tv"])
            except ValueError:
                continue
            if (
                ab_bounds[0] <= ab <= ab_bounds[1]
                and rt_bounds[0] <= rt <= rt_bounds[1]
                and tm_bounds[0] <= tm <= tm_bounds[1]
                and tv_bounds[0] <= tv <= tv_bounds[1]
            ):
                matches.append(idx)
    print(f"get_rows_within_coordinates took {time.time()-start:.3f}s")
    return matches


def update_elem_file_quick(
    elem_filename, row_indices, region_id, output_filename=None, reset_all=False
):
    start = time.time()
    row_indices = set(row_indices)
    in_place = output_filename is None
    if in_place:
        tmp = tempfile.NamedTemporaryFile("w", delete=False)
        output_filename = tmp.name
        tmp.close()
    with open(elem_filename) as infile, open(output_filename, "w") as out:
        header = next(infile, None)
        out.write(header)
        for line in infile:
            parts = line.strip().split()
            if not parts:
                out.write("\n")
                continue
            if reset_all:
                parts[-1] = "0"
            else:
                if any(int(tok) in row_indices for tok in parts[1:-1]):
                    parts[-1] = str(region_id)
            out.write(" ".join(parts) + "\n")
    if in_place:
        os.replace(output_filename, elem_filename)
    print(f"update_elem_file took {time.time()-start:.3f}s")
    return output_filename
