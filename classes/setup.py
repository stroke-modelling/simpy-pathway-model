"""
Class for organising paths, directory and file names.
"""
import os  # For defining paths.
import yaml


class Setup(object):
    """
    Global directory and file names for the pathway.

    All attributes are defined in __init__.
    The output directory attributes can be changed later using
    the method create_output_dir.
    """

    def __init__(self, *initial_data, **kwargs):

        # Directories:
        # (don't include slashes please)
        self.dir_data = 'data'
        self.dir_data_geojson = 'data_geojson'
        self.dir_output_this_setup = 'output'
        self.dir_output_all_runs = 'output_group'
        self.dir_output_combined = 'combined'
        self.dir_output = 'run'
        # Keep a list of output directories, e.g. one directory for
        # each scenario:
        self.list_dir_output = []

        # Input file names:
        self.file_input_regions = 'regions_ew.csv'
        self.file_input_unit_services = 'stroke_units_regions.csv'
        self.file_input_travel_times = 'lsoa_travel_time_matrix_calibrated.csv'
        self.file_input_travel_times_inter_unit = (
            'inter_hospital_time_calibrated.csv')
        self.file_input_lsoa_regions = 'regions_lsoa_ew.csv'
        self.file_input_hospital_info = 'stroke_hospitals_2022_regions.csv'
        self.file_input_admissions = 'admissions_2017-2019.csv'
        # Geojson files:
        self.file_geojson_lsoa = 'LSOA_(Dec_2011)_Boundaries_Super_Generalised_Clipped_(BSC)_EW_V3.geojson'
        self.file_geojson_sibcl = 'SICBL_JUL_2022_EN_BUC_4104971945004813003.geojson'
        # self.file_geojson_icb = 'ICB_JUL_2022_EN_BGC_V3_7901616774526941461.geojson'
        self.file_geojson_lhb = 'Local_Health_Boards_April_2020_WA_BGC_2022_94310626700012506.geojson'

        # Output file names:
        # Units():
        # self.file_national_unit_services = 'national_unit_services.csv'
        self.file_national_lsoa_travel = 'national_lsoa_travel_units.csv'
        self.file_national_transfer_units = 'national_transfer_units.csv'
        # Scenario():
        self.file_selected_regions = 'selected_regions.csv'
        # self.file_unit_services = 'unit_services.csv'
        self.file_selected_stroke_units = 'selected_stroke_units.csv'
        self.file_selected_transfer_units = 'selected_transfer_units.csv'
        # self.file_selected_unit_regions = 'selected_unit_regions.csv'
        self.file_selected_lsoas = None  # This gets updated later.
        # self.file_selected_lsoa_regions = 'selected_lsoa_regions.csv'
        self.file_selected_lsoa_by_catchment = 'selected_lsoa_by_catchment.csv'
        self.file_selected_lsoa_by_region_island = 'selected_lsoa_by_region_island.csv'
        self.file_selected_regions = 'selected_regions.csv'
        # Model():
        self.file_results_all = 'results_all.csv'
        self.file_results_summary_all = 'results_summary_all.csv'
        self.file_results_summary_by_admitting_unit = (
            'results_summary_by_admitting_unit.csv')
        self.file_results_summary_by_lsoa = (
            'results_summary_by_lsoa.csv')
        # Combined:
        self.file_combined_selected_regions = (
            'combined_selected_regions.csv')
        self.file_combined_selected_stroke_units = (
            'combined_selected_stroke_units.csv')
        self.file_combined_selected_transfer_units = (
            'combined_selected_transfer_units.csv')
        self.file_combined_selected_lsoas = 'combined_selected_lsoas.csv'
        self.file_combined_selected_regions = 'combined_selected_regions.csv'
        self.file_combined_results_summary_by_admitting_unit = (
            'combined_results_summary_by_admitting_unit.csv')
        self.file_combined_results_summary_by_lsoa = (
            'combined_results_summary_by_lsoa.csv')

        # Overwrite default values
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # Add a path to the top dir for this setup:
        self.dir_output_all_runs = os.path.join(
            self.dir_output_this_setup, self.dir_output_all_runs
        )
        self.create_new_top_dir = True
        # Does this dir already exist?
        if os.path.exists(self.dir_output_all_runs):
            # Flag to create a new top dir.
            self.rename_top_dir = True
        else:
            # Don't need to change the name.
            self.rename_top_dir = False

    def create_output_dir(
            self,
            dir_output,
            delim='!',
            path_to_dir=None,
            combined=False
            ):
        """
        Create a directory for storing the output of this run.

        If the chosen directory name already exists, add a suffix
        "!1" or similar to make a new directory name and create that
        instead.

        Choose a delimiter that wouldn't normally go into a dir
        name just to make the naming easier.

        Inputs
        ------
        dir_output          - str. Requested name of the output directory
                              for this run of the model only.
        delim               - str. A character to split off the requested
                              directory name from the suffix that this
                              function adds to it.
        """
        def _iterate_dir_suffix(dir_output, delim):
            """
            Update string for dir!{x} to dir!{x+1}.
            """
            if len(dir_output) > 0:
                if (dir_output[-1] == '/') | (dir_output[-1] == '\\'):
                    # Remove final '/' or '\'
                    dir_output = dir_output[:-1]
                else:
                    # Don't change the name.
                    pass
            else:
                # The name is just an empty string.
                pass
            # Split the dir name by every delim:
            dir_parts = dir_output.split(delim)
            # # Make a single string of everything up to the final delim:
            # For now, assume that the delimiter doesn't appear
            # elsewhere in the file name.
            dir_start = dir_parts[0]
            # If delimiter would appear elsewhere, would need something
            # like this only need a way to tell the difference between
            # a name ending _1 and ending _useful_stuff.
            # dir_start = (
            #     delim.join(dir_parts.split(delim)[:-1])
            #     if len(dir_parts) > 1 else dir_output)
            if len(dir_parts) == 1:
                # No delim in it yet. Set up the suffix.
                suffix = 1
            else:
                # Increase the number after the delim.
                suffix = dir_parts[-1]
                try:
                    suffix = int(suffix)
                    suffix += 1
                except ValueError:
                    # The final part of the directory name is not
                    # a number.
                    suffix = 1
            # Update the directory name:
            dir_output = f'{dir_start}{delim}{suffix}'
            return dir_output

        # First, do we need to make a new top directory?
        if self.rename_top_dir:
            # While the requested output folder already exists,
            # add a suffix or increase its number until there's a new name.
            dir_output_all_runs = self.dir_output_all_runs
            dir_output_top = os.path.split(dir_output_all_runs)[-1]
            while os.path.isdir(dir_output_all_runs):
                dir_output_top = _iterate_dir_suffix(
                    dir_output_top, delim)
                dir_output_all_runs = os.path.join(
                    self.dir_output_this_setup, dir_output_top)
            self.dir_output_all_runs = dir_output_all_runs
            # Update flag so this doesn't run again.
            self.rename_top_dir = False
        else:
            pass
        if self.create_new_top_dir:
            # Create top directory:
            os.mkdir(self.dir_output_all_runs)
            # Update flag so this doesn't run again.
            self.create_new_top_dir = False
        else:
            pass

        if path_to_dir is None:
            path_to_dir = self.dir_output_all_runs
            subdir = True
        else:
            subdir = False

        # Check if output folder already exists:
        dir_output_this_run = os.path.join(
            path_to_dir, dir_output)

        # While the requested output folder already exists,
        # add a suffix or increase its number until there's a new name.
        while os.path.isdir(dir_output_this_run):
            dir_output = _iterate_dir_suffix(dir_output, delim)
            dir_output_this_run = os.path.join(
                path_to_dir, dir_output)

        # Create this directory:
        os.mkdir(dir_output_this_run)

        if combined:
            # Save to self
            # (and so overwrite any name that was there before):
            self.dir_output_combined = dir_output_this_run
        elif subdir:
            # Add the output directory to the list:
            self.list_dir_output.append(dir_output_this_run)
            # Save to self
            # (and so overwrite any name that was there before):
            self.dir_output = dir_output_this_run

        # Return the name so that we can point the code
        # at this directory:
        return dir_output_this_run

    def save_to_file(self):
        """Save the variable dict as a .yml file."""
        setup_vars = vars(self)

        dir_output = self.dir_output_all_runs
        file_output = 'setup.yml'
        file_setup_vars = os.path.join(dir_output, file_output)

        with open(file_setup_vars, 'w') as f:
            yaml.dump(setup_vars, f)

    def import_from_file(self, path_to_setup_file):
        """Import a .yml file and overwrite attributes here."""
        with open(path_to_setup_file, 'r') as f:
            setup_vars_imported = yaml.safe_load(f)

        for key, val in setup_vars_imported.items():
            setattr(self, key, val)
