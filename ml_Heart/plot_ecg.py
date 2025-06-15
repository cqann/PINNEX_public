# plot_ecg_custom_names.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def generate_ecg_plot_simple(
    ecg_file_path: str,
    healthy_ecg_file_path: str = None,
    ecg_leads_to_plot: list = None,  # If provided, should specify the 12 leads
    ecg1_name: str = "Computed",  # Name for the first ECG dataset
    ecg2_name: str = "Healthy",  # Name for the second (healthy/comparison) ECG dataset
    # output_dir is no longer used for save location, but kept for potential future use or other logic
    output_dir: str = ".",
):
    """
    Generates and saves a plot of 12 ECG leads in a fixed 2-row (2x6) layout,
    optionally comparing against a second ECG dataset.
    Legend labels and plot title are customizable.
    Reads ECG data from CSV files. Infers or generates a time vector.
    Plot is saved in the same directory as ecg_file_path.

    Args:
        ecg_file_path (str): Path to the primary ECG data CSV file.
        healthy_ecg_file_path (str, optional): Path to the second (comparison) ECG data CSV file.
        ecg_leads_to_plot (list, optional): Specific list of 12 lead names to plot and their order.
                                         If None, the first 12 available plottable columns
                                         (excluding 'time') from ecg_file_path will be used.
        ecg1_name (str, optional): Name for the primary ECG dataset, used in legend and title.
                                   Defaults to "Computed".
        ecg2_name (str, optional): Name for the second (comparison) ECG dataset, used in legend and title.
                                   Defaults to "Healthy".
        output_dir (str, optional): Currently not used for determining plot save location.
                                    Plot is saved alongside ecg_file_path. Defaults to ".".


    Returns:
        str: Path to the saved plot image, or None if plotting failed.
    """

    # --- 1. Load Main ECG Data ---
    if not ecg_file_path:
        print("Error: ecg_file_path is required.")
        return None
    try:
        ecg_df = pd.read_csv(ecg_file_path)
        if ecg_df.empty:
            print(f"Error: ECG file is empty: {ecg_file_path}")
            return None
    except FileNotFoundError:
        print(f"Error: ECG file not found: {ecg_file_path}")
        return None
    except Exception as e:
        print(f"Error loading ECG file {ecg_file_path}: {e}")
        return None

    # --- 2. Determine Time Vector and Final 12 ECG Leads ---
    all_columns_from_csv = list(ecg_df.columns)
    time_vector = None
    time_col_name_used = None

    columns_to_check_for_time = list(all_columns_from_csv)
    for col_name in columns_to_check_for_time:
        if col_name.lower() == "time":
            try:
                time_vector = ecg_df[col_name].values.astype(float)
                time_col_name_used = col_name
                break
            except ValueError:
                time_vector = None
                time_col_name_used = None
                break

    if time_vector is None:
        time_vector = np.arange(len(ecg_df))

    available_leads_in_csv = [
        col for col in all_columns_from_csv if col != time_col_name_used
    ]

    final_leads_for_plot = []
    if ecg_leads_to_plot:
        valid_user_leads = [
            lead for lead in ecg_leads_to_plot if lead in available_leads_in_csv
        ]
        if len(valid_user_leads) < 12:
            print(
                f"Error: User specified `ecg_leads_to_plot`, but fewer than 12 were valid or found in the CSV. Found {len(valid_user_leads)} valid leads from your list: {valid_user_leads}"
            )
            return None
        final_leads_for_plot = valid_user_leads[:12]
    else:
        if len(available_leads_in_csv) < 12:
            print(
                f"Error: Not enough available leads in the CSV file to make a 12-lead plot. Found {len(available_leads_in_csv)}: {available_leads_in_csv}"
            )
            return None
        final_leads_for_plot = available_leads_in_csv[:12]

    # --- 3. Load Second (Comparison) ECG Data ---
    comparison_ecg_df = None  # Renamed from healthy_ecg_df for clarity
    if (
        healthy_ecg_file_path
    ):  # Parameter name kept for consistency with previous versions
        try:
            temp_df = pd.read_csv(healthy_ecg_file_path)
            if not temp_df.empty:
                comparison_ecg_df = temp_df
        except FileNotFoundError:
            print(
                f"Info: Comparison ECG file not found at {healthy_ecg_file_path}. Plotting primary ECG only."
            )
            pass  # Silently proceed if comparison file is not found
        except Exception as e:
            print(
                f"Warning: Could not load comparison ECG file {healthy_ecg_file_path}: {e}"
            )

    # --- 4. Plotting ---
    fig = None
    comparison_data_actually_plotted = (
        False  # Flag to track if any comparison data is shown
    )
    try:
        n_rows = 2
        n_cols = 6

        fig, axes = plt.subplots(
            n_rows,
            n_cols,
            figsize=(
                18,
                5.5,
            ),  # Slightly adjusted height for potentially longer titles/legends
            sharex=True,
            squeeze=False,
        )
        axes = axes.flatten()

        plot_count = 0

        for i, lead_name in enumerate(final_leads_for_plot):
            ax = axes[i]
            plot_ecg1_data = (
                lead_name in ecg_df.columns and not ecg_df[lead_name].isnull().all()
            )

            plot_ecg2_data = False
            if comparison_ecg_df is not None:
                plot_ecg2_data = (
                    lead_name in comparison_ecg_df.columns
                    and not comparison_ecg_df[lead_name].isnull().all()
                )
                if plot_ecg2_data:
                    comparison_data_actually_plotted = True

            if plot_ecg1_data or plot_ecg2_data:
                plot_count += 1
                ax.set_title(lead_name)
                ax.set_ylabel("Amplitude")
                ax.grid(True)

                max_time_len = len(time_vector)

                if plot_ecg1_data:
                    computed_vals = ecg_df[lead_name].values
                    len_to_plot = min(max_time_len, len(computed_vals))
                    ax.plot(
                        time_vector[:len_to_plot],
                        computed_vals[:len_to_plot],
                        label=ecg1_name,  # Use custom name
                        linewidth=1.2,
                    )

                if plot_ecg2_data:
                    comparison_vals = comparison_ecg_df[lead_name].values
                    len_to_plot = min(max_time_len, len(comparison_vals))
                    ax.plot(
                        time_vector[:len_to_plot],
                        comparison_vals[:len_to_plot],
                        label=ecg2_name,  # Use custom name
                        linestyle="--",
                        color="coral",
                        linewidth=1.0,
                    )

                if plot_ecg1_data or plot_ecg2_data:
                    ax.legend()
            else:
                ax.set_title(f"{lead_name} (No Data)")
                ax.axis("off")

        if plot_count > 0:
            time_axis_label = "Time (s)" if time_col_name_used else "Time (samples)"
            for k_ax_idx in range(n_cols, n_cols * n_rows):
                if k_ax_idx < len(axes) and axes[k_ax_idx].axison:
                    axes[k_ax_idx].set_xlabel(time_axis_label)

        base_input_filename = os.path.splitext(os.path.basename(ecg_file_path))[0]

        # Construct plot title using custom names
        plot_title = f"12-Lead ECG: {ecg1_name}"
        if comparison_ecg_df is not None and comparison_data_actually_plotted:
            plot_title += f" vs. {ecg2_name}"

        if plot_count == 0:
            plot_title = f"ECG Plotting: No data to display from {base_input_filename}"

        fig.suptitle(plot_title, fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # --- 5. Save the plot ---
        ecg_file_directory = os.path.dirname(os.path.abspath(ecg_file_path))

        plot_filename_suffix = f"{ecg1_name.replace(' ', '_')}.png"
        if comparison_ecg_df is not None and comparison_data_actually_plotted:
            plot_filename_suffix = (
                f"{ecg1_name.replace(' ', '_')}_vs_{ecg2_name.replace(' ', '_')}.png"
            )

        output_plot_filename = f"{base_input_filename}{plot_filename_suffix}"
        full_plot_path = os.path.join(ecg_file_directory, plot_filename_suffix)

        plt.savefig(full_plot_path)
        print(f"Plot saved to {full_plot_path}")
        return full_plot_path

    except Exception as e:
        print(f"An error occurred during plotting: {e}")

        return None
    finally:
        if fig is not None and plt.fignum_exists(fig.number):
            plt.close(fig)
