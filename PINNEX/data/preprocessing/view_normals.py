import matplotlib.pyplot as plt
import numpy as np
import os


def plot_interactive_heart_sliver():
    """
    Loads heart data, normalizes all normal vectors, filters points
    to a sliver where -0.1 <= Z <= 0.1, and creates an interactive 3D scatter plot.
    Axis limits are set based on the full original dataset.
    Points are colored by their normalized normal vectors (abs value),
    and arrows indicate normal vector directions.
    """
    file_path = "slab_coll_pts_normals.txt"

    if not os.path.exists(file_path):
        print(f"Error: Data file '{file_path}' not found.")
        print("Please make sure the file is in the same directory as the script.")
        return

    try:
        data = np.loadtxt(file_path)
    except Exception as e:
        print(f"Error loading data from '{file_path}': {e}")
        return

    # --- Sample 5% of the data ---
    sample_fraction = 0.5
    num_samples = int(data.shape[0] * sample_fraction)
    if num_samples > 0:
        data = data[np.random.choice(data.shape[0], num_samples, replace=False)]
    else:
        print("Not enough data points to sample 5%.")
        return

    points_all = data[:, :3]
    normals_all = data[:, 3:]

    if points_all.shape[0] == 0:
        print("No data points found in the file.")
        return

    # --- Normalize all normal vectors ---
    magnitudes = np.linalg.norm(normals_all, axis=1, keepdims=True)
    normalized_normals_all = np.divide(normals_all, magnitudes,
                                       out=np.zeros_like(normals_all),
                                       where=magnitudes != 0)

    # --- Determine overall axis limits from the full dataset ---
    x_min_all, x_max_all = np.min(points_all[:, 0]), np.max(points_all[:, 0])
    y_min_all, y_max_all = np.min(points_all[:, 1]), np.max(points_all[:, 1])
    z_min_all, z_max_all = np.min(points_all[:, 2]), np.max(points_all[:, 2])

    # --- Filter for the sliver: -0.1 <= Z <= 0.1 ---
    sliver_condition = (points_all[:, 2] >= -5.1) & (points_all[:, 2] <= 5.1)

    points_sliver = points_all[sliver_condition]
    normals_sliver = normalized_normals_all[sliver_condition]

    if points_sliver.shape[0] == 0:
        print("No points found in the Z-coordinate range -0.1 to 0.1.")
        # Optionally, still show an empty plot with correct axes
        # For now, we'll just return or inform the user.
        # To show empty plot:
        # fig = plt.figure(figsize=(12, 10))
        # ax = fig.add_subplot(111, projection='3d')
        # ax.set_xlim(x_min_all, x_max_all)
        # ax.set_ylim(y_min_all, y_max_all)
        # ax.set_zlim(z_min_all, z_max_all)
        # ax.set_xlabel('X Coordinate')
        # ax.set_ylabel('Y Coordinate')
        # ax.set_zlabel('Z Coordinate')
        # ax.set_title('Interactive 3D Heart Plot - Sliver (Z: -0.1 to 0.1) - No Points Found')
        # plt.show()
        return

    print(f"Total points in original dataset: {points_all.shape[0]}")
    print(f"Points in sliver (Z between -0.1 and 0.1): {points_sliver.shape[0]}")

    # Coloring scheme for the sliver: absolute value of NORMALIZED normal vector components
    colors_sliver = np.abs(normals_sliver)
    colors_sliver = np.clip(colors_sliver, 0, 1)

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot points in the sliver
    ax.scatter(points_sliver[:, 0], points_sliver[:, 1], points_sliver[:, 2],
               c=colors_sliver, marker='o', s=15, label=f'Sliver Points (Z: -0.1 to 0.1)')

    # Plot normal vectors for the sliver
    ax.quiver(
        points_sliver[:, 0], points_sliver[:, 1], points_sliver[:, 2],
        normals_sliver[:, 0], normals_sliver[:, 1], normals_sliver[:, 2],
        length=0.05,  # Adjust arrow length as needed for visibility in the sliver
        colors=colors_sliver,
        pivot='tail',
        label='Normalized Normal Vectors (Sliver)'
    )

    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Z Coordinate')
    ax.set_title(f'Interactive 3D Heart Plot - Sliver (Z: -0.1 to 0.1)')

    # --- Set axis limits to the full extent of the original data ---
    ax.set_xlim(x_min_all, x_max_all)
    ax.set_ylim(y_min_all, y_max_all)
    ax.set_zlim(z_min_all, z_max_all)

    # Optional: To make the visual scale of axes more uniform based on data ranges
    # This helps in making the "sliver" appear in correct proportion.
    # ax.set_box_aspect((np.ptp(points_all[:,0]), np.ptp(points_all[:,1]), np.ptp(points_all[:,2])))
    # For a 1:1:1 aspect ratio, ensuring the data is centered and limits are appropriate:
    # Get the ranges for each axis from the full dataset
    range_x = x_max_all - x_min_all
    range_y = y_max_all - y_min_all
    range_z = z_max_all - z_min_all
    max_range = max(range_x, range_y, range_z)

    mid_x = (x_max_all + x_min_all) / 2
    mid_y = (y_max_all + y_min_all) / 2
    mid_z = (z_max_all + z_min_all) / 2

    # Comment these out if you prefer the direct min/max limits set earlier
    ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
    ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
    ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

    ax.legend()
    ax.view_init(elev=20, azim=45)  # Adjust initial view if needed
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except IOError:
        print("Seaborn style 'seaborn-v0_8-darkgrid' not found, using a fallback.")
        plt.style.use('dark_background')

    print("Displaying interactive plot of the sliver... Close the plot window to exit.")
    plt.show()


if __name__ == '__main__':
    plot_interactive_heart_sliver()
