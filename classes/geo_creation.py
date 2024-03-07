"""
Draw some maps using output files.

crs reference:
+ EPSG:4326  - longitude / latitude.
+ CRS:84     - same as EPSG:4326.
+ EPSG:27700 - British National Grid (BNG).

NOTES
# Always load all data every time.
# Only load it once.
# Select different DataFrame columns for different plots.

Example catchment areas:
The two right-hand units are in the selected regions
and the two units on the left are national units,
not modelled directly.

    ▓▓▓▓▓▓▓▓▓░░░░░░░░█████▒▒▒▒▒  <-- Drip-and-ship   +------------+
    ▏   *        o     o   *  ▕                      | * MT unit  |
    ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒▒  <-- Mothership      | o IVT unit |
                -->▏ ▕<--                            +------------+
                Difference

The catchment area boundaries are halfway between adjacent units.

In the mothership model, the right-hand MT unit's catchment area
covers the whole right half of the rectangle.
In the drip-and-ship model, some of that area is instead assigned
to the out-of-area IVT unit in the centre of the rectangle.

TO DO - currently brute forcing all of the dataframes to have mathcing
column index labels and levels and that, but should be able to function
it up. Automatically know what levels are already there and which need
adding.

TO DO - automatic repr() for making maps

TO DO - better colour assignment.
e.g. drip and ship catchment map, make all feeder units different shades
of the same colour for each MT unit.

TO DO - get this ready for packaging.
"""
import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import os

from shapely import LineString  # For creating line geometry.
from shapely.geometry import Polygon # For extent box.
import string  # For generating labels for stroke units.

from classes.setup import Setup
import classes.map_functions as maps  # for plotting.


# ##########################
# ##### FILE SELECTION #####
# ##########################
def set_data_dir(self, dir_data):
    """
    On changing dir, wipe the loaded data and reload from this dir.
    """
    combo_condition = (
        (dir_data is None) |
        (dir_data == 'combined') |
        (dir_data == self.setup.dir_output_combined)
        )
    if combo_condition:
        # Use combined data files.
        self.dir_data = self.setup.dir_output_combined
        self.data_type = 'combined'
        self.dir_output_maps = os.path.join(
            self.dir_data, self.setup.name_dir_output_maps
        )
    else:
        # Check that we have the most up-to-date list of dirs:
        self.setup.make_list_dir_scenario()
        # Use files for the selected scenario only.
        for d, dir_scen in enumerate(self.setup.list_dir_scenario):
            end = os.path.split(dir_scen)[-1]
            if end == dir_data:
                self.dir_data = dir_scen
        # Add the "pathway" and "maps" parts:
        self.dir_output_maps = os.path.join(
            self.dir_data, self.setup.name_dir_output_maps
        )
        self.dir_data = os.path.join(
            self.dir_data,
            self.setup.name_dir_output_pathway
            )
        self.data_type = 'single'

    self.delete_loaded_data()

    # Create a new maps/ dir for outputs.
    try:
        os.mkdir(self.dir_output_maps)
    except FileExistsError:
        # The directory already exists.
        pass


def delete_loaded_data(self):
    # Delete these attributes if they exist:
    data_attrs = [
        # Loaded in from file:
        'df_regions',
        'df_units',
        'df_transfer',
        'df_lsoa',
        'df_results_by_unit',
        'df_results_by_lsoa',
        # Combined data from files:
        'gdf_boundaries_regions',
        'gdf_points_units',
        'gdf_lines_transfer',
        'gdf_boundaries_lsoa',
    ]
    for attr in data_attrs:
        try:
            delattr(self, attr)
        except AttributeError:
            # The data wasn't loaded anyway.
            pass


def load_run_data(self, load_list=[]):
    """
    Load in data specific to these runs.

    TO DO - reasonably expect that the selected regions and units files might change
    after being plotted once, so need a way to force reload of the data.

    TO DO - what if no setup?
    """
    # Setup for combined files:
    dicts_combo = {
        'df_regions': {
            'file': self.setup.file_combined_selected_regions,
            'header': [0, 1],
            'index_col': [0, 1],
            },
        'df_units': {
            'file': self.setup.file_combined_selected_units,
            'header': [0, 1],
            'index_col': 0,
            },
        'df_transfer': {
            'file': self.setup.file_combined_selected_transfer_units,
            'header': [0, 1],
            'index_col': [0, 1],
            },
        'df_lsoa': {
            'file': self.setup.file_combined_selected_lsoas,
            'header': [0, 1],
            'index_col': 1,
            },
        'df_results_by_unit': {
            'file': (
                self.setup.file_combined_results_summary_by_admitting_unit
                ),
            'header': [0, 1, 2],
            'index_col': 0,
            },
        'df_results_by_lsoa': {
            'file': self.setup.file_combined_results_summary_by_lsoa,
            'header': [0, 1, 2],
            'index_col': 1,
            },
    }
    # Setup for individual run's files:
    dicts_single = {
        'df_regions': {
            'file': self.setup.file_selected_regions,
            'header': [0],
            'index_col': [0, 1],
            },
        'df_units': {
            'file': self.setup.file_selected_units,
            'header': [0],
            'index_col': 0,
            },
        'df_transfer': {
            'file': self.setup.file_selected_transfer_units,
            'header': [0],
            'index_col': 0,
            },
        'df_lsoa': {
            'file': self.setup.file_selected_lsoas,
            'header': [0],
            'index_col': 1,
            },
        'df_results_by_unit': {
            'file': (
                self.setup.file_results_summary_by_admitting_unit
                ),
            'header': [0, 1],
            'index_col': 0,
            },
        'df_results_by_lsoa': {
            'file': self.setup.file_results_summary_by_lsoa,
            'header': [0, 1],
            'index_col': 1,
            },
    }
    if self.data_type == 'combined':
        dicts_data = dicts_combo
    else:
        dicts_data = dicts_single

    if len(load_list) == 0:
        # Load everything.
        load_list = list(dicts_data.keys())

    for label in load_list:
        data_dict = dicts_data[label]
        # Make path to file:
        try:
            path_to_file = os.path.join(self.dir_data, data_dict['file'])
        except TypeError:
            # The file name is None.
            # Can't import this.
            path_to_file = ''
        try:
            # Specify header to import as a multiindex DataFrame.
            df = pd.read_csv(
                path_to_file,
                header=data_dict['header'],
                index_col=data_dict['index_col']
                )
            if ((label == 'df_transfer') & (self.data_type == 'single')):
                # Add another column to the index.
                # Assume that the wanted column isn't in slot 1
                # and its location isn't known, so call it by name.
                iname = df.index.name
                df = df.reset_index()
                df = df.set_index([iname, 'name_nearest_mt'])
            else:
                pass
            # Save to self:
            setattr(self, label, df)
        except FileNotFoundError:
            raise FileNotFoundError(
                f'Cannot import {label} from {data_dict["file"]}'
                ) from None


def process_data(self, load_list=[]):
    """
    Load everything in mmkay  TO DO - write me
    """
    def _check_prereqs_and_load(prereqs, func):
        data_loaded = [hasattr(self, df) for df in prereqs]
        if all(data_loaded):
            func()
        else:
            # Something is missing so can't load the data.
            pass

    if 'gdf_boundaries_regions' in load_list:
        func = self._load_geometry_regions
        prereqs = ['df_regions']
        _check_prereqs_and_load(prereqs, func)
    if 'gdf_points_units' in load_list:
        func = self._load_geometry_stroke_units
        prereqs = ['df_regions', 'df_units']
        _check_prereqs_and_load(prereqs, func)
    if 'gdf_lines_transfer' in load_list:
        func = self._load_geometry_transfer_units
        prereqs = ['df_regions', 'df_units', 'df_transfer']
        _check_prereqs_and_load(prereqs, func)
    if 'gdf_boundaries_lsoa' in load_list:
        func = self._load_geometry_lsoa
        prereqs = ['df_regions', 'df_lsoa']
        _check_prereqs_and_load(prereqs, func)


def _check_prereqs_exist(self, prereqs):
    """
    Run this at start of plot setup functions.
    """
    missing_attrs = []
    for attr in prereqs:
        if hasattr(self, attr):
            pass
        else:
            missing_attrs.append(attr)
    if len(missing_attrs) > 0:
        err = ''.join([
            f'Missing some information: {missing_attrs}\n',
            'Try reloading: self.load_run_data() and ',
            'self.process_data()'
        ])
        raise Exception(err) from None
    else:
        pass


def import_geojson(self, region_type: 'str'):
    """
    Import a geojson file as GeoDataFrame.

    The crs (coordinate reference system) is set to British National
    Grid.

    Inputs
    ------
    setup       - Setup() object. Contains attributes for paths to the
                    data directory and the geojson file names.
    region_type - str. Lookup name for selecting a geojson file.
                    This should be one of the column names from the
                    various regions files.

    Returns
    -------
    gdf_boundaries - GeoDataFrame. One row per region shape in the
                    file. Expect columns for region name and geometry.
    """
    # Select geojson file based on input region type:
    geojson_file_dict = {
        'LSOA11NM': self.setup.file_geojson_lsoa,
        'SICBL22NM': self.setup.file_geojson_sibcl,
        'LHB20NM': self.setup.file_geojson_lhb,
    }

    # Import region file:
    dir_input = self.setup.dir_reference_data_geojson
    file_input = geojson_file_dict[region_type]
    path_to_file = os.path.join(dir_input, file_input)
    gdf_boundaries = geopandas.read_file(path_to_file)

    if region_type == 'LSOA11NM':
        index_col = 'LSOA11CD'
        # Only keep these columns.
        # Don't keep LSOA11NM because that will be merged in later
        # from an LSOA dataframe.
        geo_cols = ['BNG_E', 'BNG_N',
                    'LONG', 'LAT', 'GlobalID', 'geometry']

    else:
        index_col = 'region_code'
        # Only keep these columns:
        geo_cols = ['region', 'BNG_E', 'BNG_N',
                    'LONG', 'LAT', 'GlobalID', 'geometry']

        # Find which columns to rename to 'region' and 'region_code'.
        if (region_type.endswith('NM') | region_type.endswith('NMW')):
            region_prefix = region_type.removesuffix('NM')
            region_prefix = region_prefix.removesuffix('NMW')
            region_code = region_prefix + 'CD'
        elif (region_type.endswith('nm') | region_type.endswith('nmw')):
            region_prefix = region_type.removesuffix('NM')
            region_prefix = region_prefix.removesuffix('NMW')
            region_code = region_prefix + 'cd'
        else:
            # This shouldn't happen.
            # TO DO - does this need a proper exception or can
            # we just change the above to if/else? ------------------------------
            region_code = region_type[:-2] + 'CD'

        try:
            # Rename this column:
            gdf_boundaries = gdf_boundaries.rename(columns={
                region_type: 'region',
                region_code: 'region_code'
            })
        except KeyError:
            # That column doesn't exist.
            # Try finding a column that has the same start and end
            # as requested:
            prefix = region_type[:3]
            suffix = region_type[-2:]
            success = False
            for column in gdf_boundaries.columns:
                # Casefold turns all UPPER into lower case.
                match = ((column[:3].casefold() == prefix.casefold()) &
                            (column[-2:].casefold() == suffix.casefold()))
                if match:
                    # Rename this column:
                    col_code = column[:-2] + region_code[-2:]
                    gdf_boundaries = gdf_boundaries.rename(columns={
                        column: 'region',
                        col_code: 'region_code'
                        })
                    success = True
                else:
                    pass
            if success is False:
                pass
                # TO DO - proper error here --------------------------------

    # Set the index:
    gdf_boundaries = gdf_boundaries.set_index(index_col)
    # Only keep geometry data:
    gdf_boundaries = gdf_boundaries[geo_cols]

    # If crs is given in the file, geopandas automatically
    # pulls it through. Convert to National Grid coordinates:
    if gdf_boundaries.crs != 'EPSG:27700':
        gdf_boundaries = gdf_boundaries.to_crs('EPSG:27700')
    return gdf_boundaries

# ##########################
# ##### DATA WRANGLING #####
# ##########################
def _load_geometry_regions(
        self, limit_to_england=False, limit_to_wales=False):
    """
    Create GeoDataFrames of new geometry and existing DataFrames.
    """
    # ----- Gather data -----
    # Selected regions names and usage.
    # ['{region type}', 'selected', 'contains_selected_lsoa']
    df_regions = self.df_regions
    # Index column: 'region'.
    # Expected column MultiIndex levels:
    #   - combined: ['scenario', 'property']
    #   - separate: ['{unnamed level}']

    # All region polygons:
    gdf_list = []
    if limit_to_wales is False:
        gdf_boundaries_regions_e = self.import_geojson('SICBL22NM')
        gdf_list.append(gdf_boundaries_regions_e)
    if limit_to_england is False:
        gdf_boundaries_regions_w = self.import_geojson('LHB20NM')
        gdf_list.append(gdf_boundaries_regions_w)
    if len(gdf_list) > 1:
        # Combine:
        gdf_boundaries_regions = pd.concat(gdf_list, axis='rows')
    elif len(gdf_list) == 0:
        # Haven't loaded any boundaries.
        # Probably don't want this to happen...
        err = 'No boundaries loaded - England and Wales both excluded.'
        raise Exception(err) from None

    # Index column: 'region'.
    # Always has only one unnamed column index level.

    # Drop columns that appear in both DataFrames:
    gdf_boundaries_regions = gdf_boundaries_regions.drop(
        'region', axis='columns'
    )

    # ----- Prepare separate data -----
    # Set up column level info for the merged DataFrame.
    # The "combined" scenario Dataframe will have an extra
    # column level with the scenario name.
    if self.data_type == 'combined':
        # Everything needs two levels: scenario, property.
        # Geometry:
        cols_gdf_boundaries_regions = [
            ['any'] * len(gdf_boundaries_regions.columns),  # scenario
            gdf_boundaries_regions.columns,                 # property
        ]
        # Final data:
        col_level_names = ['scenario', 'property']
        col_geometry = ('any', 'geometry')
    else:
        # Everything needs one level: property.
        # Geometry:
        cols_gdf_boundaries_regions = gdf_boundaries_regions.columns
        # Final data:
        col_level_names = ['property']
        col_geometry = 'geometry'

    # Geometry:
    gdf_boundaries_regions = pd.DataFrame(
        gdf_boundaries_regions.values,
        index=gdf_boundaries_regions.index,
        columns=cols_gdf_boundaries_regions
    )

    # ----- Create final data -----
    # Merge together the DataFrames.
    gdf_boundaries_regions = pd.merge(
        gdf_boundaries_regions, df_regions,
        left_index=True, right_index=True, how='right'
    )

    # Name the column levels:
    gdf_boundaries_regions.columns = (
        gdf_boundaries_regions.columns.set_names(col_level_names))

    if self.data_type == 'combined':
        # Sort the results by scenario (top column index):
        cols_scen = (
            gdf_boundaries_regions.columns.get_level_values('scenario'))
        cols_scen = sorted(list(set(cols_scen)))
        gdf_boundaries_regions = gdf_boundaries_regions[cols_scen]
    else:
        # Don't sort.
        pass

    # Convert to GeoDataFrame:
    gdf_boundaries_regions = geopandas.GeoDataFrame(
        gdf_boundaries_regions,
        geometry=col_geometry
        )

    # ----- Save to self -----
    self.gdf_boundaries_regions = gdf_boundaries_regions

    # Save output to output folder.
    dir_output = self.dir_output_maps
    file_name = self.setup.file_gdf_boundaries_regions
    path_to_file = os.path.join(dir_output, file_name)
    gdf_boundaries_regions.to_csv(path_to_file)


def _load_geometry_stroke_units(self):
    """
    Create GeoDataFrames of new geometry and existing DataFrames.

    Import GeoDataFrame of selected stroke unit data.

    The file read is the output from Scenario() after the
    national stroke units have been reduced to those selected.

    The file contains coordinates (Easting, Northing) and (long, lat)
    which are picked out here and converted into Point() objects.
    The crs (coordinate reference system) is set to British National
    Grid.

    Inputs
    ------
    setup - Setup() object. Contains attributes for paths to the
            data directory and the geojson file names.

    Returns
    -------
    gdf_units - GeoDataFrame. One row per selected stroke unit in
                the file. Columns include unit name and geometry.
    """
    # Selected stroke units names, services, and regions.
    df_units = self.df_units
    # Index column: Postcode.
    # Expected column MultiIndex levels:
    #   - combined: ['scenario', 'property']
    #   - separate: ['{unnamed level}']

    # Load and parse geometry data
    dir_input = self.setup.dir_reference_data
    file_input = self.setup.file_input_hospital_coords
    path_to_file = os.path.join(dir_input, file_input)
    df_coords = pd.read_csv(path_to_file, index_col='postcode')
    # Index: postcode.
    # Columns: BNG_E, BNG_N, Longitude, Latitude.
    if self.data_type == 'combined':
        # Add another column level to the coordinates.
        cols_df_coords = [
            ['any'] * len(df_coords.columns),  # scenario
            df_coords.columns,                 # property
        ]
        df_coords = pd.DataFrame(
            df_coords.values,
            index=df_coords.index,
            columns=cols_df_coords
        )
    # Merge:
    df_units = pd.merge(
        df_units, df_coords, left_index=True, right_index=True, how='left')

    if self.data_type == 'combined':
        x_col = ('any', 'BNG_E')
        y_col = ('any', 'BNG_N')
        coords_col = ('any', 'geometry')
    else:
        x_col = 'BNG_E'
        y_col = 'BNG_N'
        coords_col = 'geometry'

    # Convert to geometry (point):
    # Create coordinates:
    # Current setup means sometimes these columns have different names.
    # TO DO - fix that please! ---------------------------------------------------
    # Extra .values.reshape are to remove the column headings.
    x = df_units[x_col].values.reshape(len(df_units))
    y = df_units[y_col].values.reshape(len(df_units))
    crs = 'EPSG:27700'  # by definition for easting/northing.

    # Convert each pair of coordinates to a Point(x, y).
    df_units[coords_col] = geopandas.points_from_xy(x, y)

    # Convert to GeoDataFrame:
    gdf_units = geopandas.GeoDataFrame(
        df_units, geometry=coords_col, crs=crs)

    self.gdf_points_units = gdf_units

    # Save output to output folder.
    dir_output = self.dir_output_maps
    file_name = self.setup.file_gdf_points_units
    path_to_file = os.path.join(dir_output, file_name)
    gdf_units.to_csv(path_to_file)


def _load_geometry_transfer_units(self):
    """
    Create GeoDataFrames of new geometry and existing DataFrames.
    """
    # Selected stroke units names, coordinates, and services.
    df_transfer = self.df_transfer
    # Index column: ['postcode', 'name_nearest_mt']
    # Expected column MultiIndex levels:
    #   - combined: ['scenario', 'property']
    #   - separate: ['{unnamed level}']

    # Load and parse geometry data
    dir_input = self.setup.dir_reference_data
    file_input = self.setup.file_input_hospital_coords
    path_to_file = os.path.join(dir_input, file_input)
    df_coords = pd.read_csv(path_to_file)
    # Columns: postcode, BNG_E, BNG_N, Longitude, Latitude.

    if self.data_type == 'combined':
        # From the loaded file:
        x_col = ('any', 'BNG_E')
        y_col = ('any', 'BNG_N')
        x_col_mt = ('any', 'BNG_E_mt')
        y_col_mt = ('any', 'BNG_N_mt')
        # To be created here:
        col_unit = ('any', 'unit_coords')
        col_tran = ('any', 'transfer_coords')
        col_line_coords = ('any', 'line_coords')
        col_line_geometry = ('any', 'geometry')
    else:
        # From the loaded file:
        x_col = 'BNG_E'
        y_col = 'BNG_N'
        x_col_mt = 'BNG_E_mt'
        y_col_mt = 'BNG_N_mt'
        # To be created here:
        col_unit = 'unit_coords'
        col_tran = 'transfer_coords'
        col_line_coords = 'line_coords'
        col_line_geometry = 'geometry'

    # DataFrame of just the arrival and transfer units:
    df_arrival_transfer = df_transfer.index.to_frame(index=False)
    # If there are multiple column levels, only keep the lowest.
    if df_arrival_transfer.columns.nlevels > 1:
        # TO DO - drop by level name. --------------------------------------------------
        df_arrival_transfer = (
            df_arrival_transfer.droplevel(0, axis='columns'))
    # Index: {generic numbers}
    # Columns: 'from_postcode', 'name_nearest_mt'

    # Merge in the arrival unit coordinates:
    m1 = pd.merge(
        df_arrival_transfer, df_coords,
        left_on='postcode', right_on='postcode',
        how='left'
        )
    m2 = pd.merge(
        m1, df_coords,
        left_on='name_nearest_mt', right_on='postcode',
        how='left', suffixes=(None, '_mt')
        )
    df_arrival_transfer = m2.drop(['postcode_mt'], axis='columns')
    # Set the index columns to match the main DataFrame's:
    df_arrival_transfer = df_arrival_transfer.set_index(
        ['postcode', 'name_nearest_mt'])
    # Index: 'postcode', 'name_nearest_mt'
    # Columns: BNG_E, BNG_N, Longitude, Latitude,
    #          BNG_E_mt, BNG_N_mt, Longitude_mt, Latitude_mt

    if self.data_type == 'combined':
        # Add another column level to the coordinates.
        cols_df_arrival_transfer = [
            ['any'] * len(df_arrival_transfer.columns),  # scenario
            df_arrival_transfer.columns,                 # property
        ]
        df_arrival_transfer = pd.DataFrame(
            df_arrival_transfer.values,
            index=df_arrival_transfer.index,
            columns=cols_df_arrival_transfer
        )

    # Merge this into the main DataFrame:
    df_transfer = pd.merge(
        df_transfer, df_arrival_transfer,
        left_index=True, right_index=True, how='left')

    # Make a column of coordinates [x, y]:
    xy = df_transfer[[x_col, y_col]]
    df_transfer[col_unit] = xy.values.tolist()

    xy_mt = df_transfer[[x_col_mt, y_col_mt]]
    df_transfer[col_tran] = xy_mt.values.tolist()

    # Convert to geometry (line).
    gdf_transfer = self.create_lines_from_coords(
        df_transfer,
        [col_unit, col_tran],
        col_line_coords,
        col_line_geometry
        )

    self.gdf_lines_transfer = gdf_transfer

    # Save output to output folder.
    dir_output = self.dir_output_maps
    file_name = self.setup.file_gdf_lines_transfer
    path_to_file = os.path.join(dir_output, file_name)
    gdf_transfer.to_csv(path_to_file)


def _load_geometry_lsoa(self):
    """
    Create GeoDataFrames of new geometry and existing DataFrames.
    """
    # ----- Gather data -----
    # Selected LSOA names, codes.
    # ['lsoa', 'LSOA11CD', 'postcode_nearest', 'time_nearest', 'Use']
    df_lsoa = self.df_lsoa
    # Index column: LSOA11NM.
    # Expected column MultiIndex levels:
    #   - combined: ['scenario', 'property']
    #   - separate: ['{unnamed level}']

    # All LSOA shapes:
    gdf_boundaries_lsoa = self.import_geojson('LSOA11NM')
    # Index column: LSOA11CD.
    # Always has only one unnamed column index level.

    if hasattr(self, 'df_regions'):
        # TO DO - shorten LSOA geojson load time by separating big file into multiple smaller ones,
        # e.g. one file per region of the UK (South West, West Midlands...)
        # or one file per SICBL / LHB (too many files?)
        # then work out which ones should be loaded and only load those ones.
        # Will almost never want to load in Newcastle LSOA when we're looking at Cornwall.
        pass
    else:
        # Just load in the whole lot.
        pass

    # Results by LSOA.
    results_exist = False
    try:
        # If the file wasn't loaded, this gives AttributeError:
        df_lsoa_results = self.df_results_by_lsoa
        # Index column: lsoa.
        # Expected column MultiIndex levels:
        #   - combined: ['scenario', 'property', 'subtype']
        #   - separate: ['property', 'subtype]
        results_exist = True
    except AttributeError:
        # Continue without the results data.
        # Some plots don't need it, e.g. LSOA catchment areas.
        pass

    # ----- Prepare separate data -----
    # Set up column level info for the merged DataFrame.
    # The "combined" scenario Dataframe will have an extra
    # column level with the scenario name.
    if self.data_type == 'combined':
        # Everything needs three levels: scenario, property, subtype.
        # LSOA names:
        cols_lsoa = df_lsoa.columns
        df_lsoa_column_arr = np.array(
            [[n for n in c] for c in cols_lsoa])
        cols_df_lsoa = [
            df_lsoa_column_arr[:, 0],                    # scenario
            df_lsoa_column_arr[:, 1],                    # property
            [''] * len(cols_lsoa),                       # subtype
        ]
        # Geometry:
        cols_gdf_boundaries_lsoa = [
            ['any'] * len(gdf_boundaries_lsoa.columns),  # scenario
            gdf_boundaries_lsoa.columns,                 # property
            [''] * len(gdf_boundaries_lsoa.columns),     # subtype
        ]
        # Final data:
        col_level_names = ['scenario', 'property', 'subtype']
        col_geometry = ('any', 'geometry', '')
    else:
        # Everything needs two levels: property, subtype.
        # LSOA names:
        cols_lsoa = df_lsoa.columns
        cols_df_lsoa = [
            cols_lsoa,                                   # property
            [''] * len(cols_lsoa),                       # subtype
        ]
        # Geometry:
        cols_gdf_boundaries_lsoa = [
            gdf_boundaries_lsoa.columns,                 # property
            [''] * len(gdf_boundaries_lsoa.columns),     # subtype
        ]
        # Final data:
        col_level_names = ['property', 'subtype']
        col_geometry = ('geometry', '')

    # Make all data to be combined have the same column levels.
    # LSOA names:
    df_lsoa = pd.DataFrame(
        df_lsoa.values, index=df_lsoa.index, columns=cols_df_lsoa)

    # Geometry:
    gdf_boundaries_lsoa = pd.DataFrame(
        gdf_boundaries_lsoa.values,
        index=gdf_boundaries_lsoa.index,
        columns=cols_gdf_boundaries_lsoa
    )

    # Results:
    # This already has the right number of column levels.

    # ----- Create final data -----
    # Merge together all of the DataFrames.
    gdf_boundaries_lsoa = pd.merge(
        gdf_boundaries_lsoa, df_lsoa,
        left_index=True, right_index=True, how='right'
    )
    if results_exist:
        gdf_boundaries_lsoa = pd.merge(
            gdf_boundaries_lsoa, df_lsoa_results,
            left_index=True, right_index=True, how='left'
        )
    # Name the column levels:
    gdf_boundaries_lsoa.columns = (
        gdf_boundaries_lsoa.columns.set_names(col_level_names))

    if self.data_type == 'combined':
        # Sort the results by scenario (top column index):
        cols_scen = (
            gdf_boundaries_lsoa.columns.get_level_values('scenario'))
        cols_scen = sorted(list(set(cols_scen)))
        gdf_boundaries_lsoa = gdf_boundaries_lsoa[cols_scen]
    else:
        # Don't sort.
        pass

    # Convert to GeoDataFrame:
    gdf_boundaries_lsoa = geopandas.GeoDataFrame(
        gdf_boundaries_lsoa,
        geometry=col_geometry
        )

    # ----- Save to self -----
    self.gdf_boundaries_lsoa = gdf_boundaries_lsoa

    # Save output to output folder.
    dir_output = self.dir_output_maps
    file_name = self.setup.file_gdf_boundaries_lsoa
    path_to_file = os.path.join(dir_output, file_name)
    gdf_boundaries_lsoa.to_csv(path_to_file)


def _remove_excess_heading_from_gdf(
        self, gdf, level_to_drop, col_geometry):
    """
    TO DO - write me.
    """
    gdf = gdf.droplevel(level_to_drop, axis='columns')
    # The geometry column is still defined with the excess
    # heading, so update which column is geometry:
    gdf = gdf.set_geometry(col_geometry)
    return gdf

# ############################
# ##### HELPER FUNCTIONS #####
# ############################
def create_lines_from_coords(
        self,
        df,
        cols_with_coords,
        col_coord,
        col_geom
        ):
    """
    Convert DataFrame with coords to GeoDataFrame with LineString.

    Initially group coordinates from multiple columns into one:
    +--------+--------+       +------------------+
    |  col1  |  col2  |       |   line_coords    |
    +--------+--------+  -->  +------------------+
    | [a, b] | [c, d] |       | [[a, b], [c, d]] |
    +--------+--------+       +------------------+
    And convert the single column into shapely.LineString objects
    with associated crs. Then convert the input DataFrame into
    a GeoDataFrame with the new Line objects.

    Inputs
    ------
    df               - pd.DataFrame. Contains some coordinates.
    cols_with_coords - list. List of column names in df that contain
                    coordinates for making lines.

    Returns
    -------
    gdf - GeoDataFrame. The input df with the new Line
        geometry objects and converted to a GeoDataFrame.
    """
    # Combine multiple columns of coordinates into a single column
    # with a list of lists of coordinates:
    df[col_coord] = df[cols_with_coords].values.tolist()

    # Drop any duplicates:
    df = df.drop_duplicates(col_coord)

    # Convert line coords to LineString objects:
    df[col_geom] = [LineString(coords) for coords in df[col_coord].values]

    # Convert to GeoDataFrame:
    gdf = geopandas.GeoDataFrame(df, geometry=col_geom)  #, crs="EPSG:4326"
    # if isinstance(col_geom, tuple):
    #     gdf['geometry']
    # TO DO - implement CRS explicitly ---------------------------------------------
    return gdf


def create_combo_cols(self, gdf, scenario):
    """
    TO DO - write me
    When dataframe doesn't have the diff_this_minus_that columns,
    use this function to create that data and prevent KeyError later.

    TO DO - currently the combo column takes the max of both.
    This is good for stroke units (yes in drip and ship vs no in mothership)
    but bad for regions catching LSOA (outcome diff is NaN when not in both,
    so the regions contain no info).
    Allow selection of min and max. (Or anything else?)
    """
    # Find out what diff what:
    scen_bits = scenario.split('_')
    # Assume scenario = diff_scenario1_minus_scenario2:
    scen1 = scen_bits[1]
    scen2 = scen_bits[3]

    cols_to_combine = gdf[scen1].columns.to_list()
    for col in cols_to_combine:
        if isinstance(col, tuple):
            scen_col = (scenario, *col)
            scen1_col = (scen1, *col)
            scen2_col = (scen2, *col)
        else:
            # Assume it's a string.
            scen_col = (scenario, col)
            scen1_col = (scen1, col)
            scen2_col = (scen2, col)
        gdf[scen_col] = gdf[[scen1_col, scen2_col]].max(axis='columns')
    return gdf


def assign_colours_to_regions(self, gdf, region_type, col_col):
    """
    wip, this version pretty useless.
    """

    colours = ['ForestGreen', 'LimeGreen', 'RebeccaPurple', 'Teal']

    # Use any old colours as debug:
    np.random.seed(42)
    colour_arr = np.random.choice(colours, size=len(gdf))

    # Add to the DataFrame:
    gdf[col_col] = colour_arr

    return gdf


def assign_colours_to_regions_BROKEN(self, gdf, region_type):

    colours = ['ForestGreen', 'LimeGreen', 'RebeccaPurple', 'Teal']

    # TO DO - this neighbours function isn't working right -----------------------------
    # currently says that Gloucestershire doesn't border Bristol or Bath regions -------
    gdf = self.find_neighbours_for_regions(gdf, self.region_type)
    gdf = gdf.sort_values('total_neighbours', ascending=False)

    neighbour_list = gdf[self.region_type].tolist()

    neighbour_grid = np.full((len(gdf), len(gdf)), False)
    for row, neighbour_list_here in enumerate(gdf['neighbour_list']):
        for n in neighbour_list_here:
            col = neighbour_list.index(n)
            neighbour_grid[row, col] = True
            neighbour_grid[col, row] = True

    # Make a grid. One column per colour, one row per region.
    colour_grid = np.full((len(gdf), len(colours)), True)
    # To index row x: colour_grid[x, :]
    # To index col x: colour_grid[:, x]

    for row, region in enumerate(neighbour_list):
        # Which colours can this be?
        colours_here = colour_grid[row, :]

        # Pick the first available colour.
        ind_to_pick = np.where(colours_here == True)[0][0]

        # Update its neighbours' colour information.
        rows_neighbours = np.where(neighbour_grid[row, :] == True)[0]
        # Only keep these rows when we haven't checked them yet:
        rows_neighbours = [r for r in rows_neighbours if r > row]
        colour_grid[rows_neighbours, ind_to_pick] = False

        # Update its own colour information.
        colour_grid[row, :] = False
        colour_grid[row, ind_to_pick] = True

    # Use the bool colour grid to assign colours:
    colour_arr = np.full(len(neighbour_list), colours[0], dtype=object)
    for i, colour in enumerate(colours):
        colour_arr[np.where(colour_grid[:, i] == True)] = colour

    # Add to the DataFrame:
    gdf['colour'] = colour_arr

    # Use any old colours as debug:
    # np.random.seed(42)
    # colour_arr = np.random.choice(colours, size=len(gdf))

    return gdf


def round_coordinates(self, df_geojson):
    from shapely.geometry import shape, mapping  # For conversion from shapely polygon to geojson and back
    # Remove floating point errors
    for i in range(len(df_geojson)):
        poly = df_geojson['geometry'][i]
        # Convert shapely object to geojson object
        gpoly = mapping(poly)
        if len(gpoly['coordinates']) == 1:
            # This is probably a normal polygon.
            a_coords = np.array(gpoly['coordinates'])
            new_coords = np.round(a_coords, 3)
        else:
            # This is probably a multipolygon but could be a polygon
            # that has multiple sets of coordinates for some reason
            # (maybe a hole, doughnut-shaped polygon?).
            new_coords = []
            for c, coords in enumerate(gpoly['coordinates']):
                a_coords = np.array(gpoly['coordinates'][c])
                a_coords = np.round(a_coords, 3)
                new_coords.append(a_coords)
        gpoly['coordinates'] = new_coords

        # Convert back to shapely object
        poly = shape(gpoly)
        # Place back into the DataFrame
        df_geojson['geometry'][i] = poly
        return df_geojson


def find_neighbours_for_regions(self, df_geojson, region_type='ICG22NM'):
    import shapely

    def split_multipolygons(df_geojson):
        # Expand the dataframe - need a separate row for each polygon, so split apart any multipolygons.
        df = df_geojson.copy()
        df_new = pd.DataFrame(columns=df.columns)

        r = 0
        for i in range(len(df)):
            row = df.iloc[i]
            if isinstance(row['geometry'], shapely.geometry.polygon.Polygon):
                # All ok here, copy row contents exactly:
                df_new.loc[r] = df.iloc[i]
                r += 1
            else:
                # MultiPolygon! Split it.
                # Place each Polygon on its own row.
                multipoly = row['geometry']
                for poly in list(multipoly.geoms):
                    row_new = row.copy()
                    row_new['geometry'] = poly
                    df_new.loc[r] = row_new
                    r += 1

        # Convert this to a GeoDataFrame to match the input df_geojson:
        df_new = geopandas.GeoDataFrame(
            df_new, geometry=df_new['geometry']#, crs="EPSG:4326"
        )
        return df_new

    def find_neighbours(df_new, region_type):
        df = df_new.copy()
        df['my_neighbors'] = [[]] * len(df)

        for index, row in df.iterrows():
            if isinstance(row['geometry'], shapely.geometry.polygon.Polygon): 
                neighbors = df[df.geometry.touches(row['geometry'])][region_type].tolist()
            elif not isinstance(row['geometry'], shapely.geometry.point.Point):
                # This is a MultiPolygon. Check each Polygon separately.
                multipoly = row['geometry']
                neighbors = []
                for polygon in list(multipoly.geoms):
                    neighbours_here = df[df.geometry.intersects(polygon)].index.tolist()
                    neighbors += neighbours_here
            else:
                # It's a point! Ignore.
                pass
            try:
                # Don't let the place be its own neighbour.
                neighbors = neighbors.remove(row.name)
            except ValueError:
                # Its own name is not in the list of neighbours.
                pass
            df.at[index, 'my_neighbors'] = neighbors
            # df.loc[index]['my_neighbors'] = neighbors #", ".join([f'{n}' for n in neighbors])

        df_neighbours = df.copy()
        return df_neighbours

    def unsplit_multipolygons(df_geojson, df_neighbours):
        df = df_geojson.copy()
        df['neighbour_list'] = [[]] * len(df)

        for i in range(len(df)):
            # What is this called in the original dataframe?
            objectid = df.iloc[i]['OBJECTID']
            # region_name = df.iloc[i]['CCG19CD']
            # Where did this end up in the polygon dataframe?
            df_here = df_neighbours[df_neighbours['OBJECTID'] == objectid]
            # Combine multiple lists of neighbours into one list:
            list_of_neighbours = np.concatenate(df_here['my_neighbors'].values)
            # Remove any repeats:
            list_of_neighbours = list(set(list_of_neighbours))
            df.at[i, 'neighbour_list'] = list_of_neighbours
        return df

    df_geojson = self.round_coordinates(df_geojson)
    df_new = split_multipolygons(df_geojson)
    df_neighbours = find_neighbours(df_new, self.region_type)
    df = unsplit_multipolygons(df_geojson, df_neighbours)

    # Record the number of neighbours:
    df['total_neighbours'] = df['neighbour_list'].str.len()
    return df


def get_selected_area_extent(
        self,
        gdf_selected,
        leeway=20000,
        ):
    """
    What is the spatial extent of everything in this GeoDataFrame?
    """
    minx, miny, maxx, maxy = gdf_selected.geometry.total_bounds
    # Give this some leeway:
    minx -= leeway
    miny -= leeway
    maxx += leeway
    maxy += leeway
    map_extent = [minx, maxx, miny, maxy]
    # Turn the points into a box:
    box = Polygon((
        (minx, miny),
        (minx, maxy),
        (maxx, maxy),
        (maxx, miny),
        (minx, miny),
    ))
    return box, map_extent


def _keep_only_geometry_in_box(self, gdf, box):
    mask = gdf.geometry.intersects(box)
    gdf = gdf[mask]
    return gdf


def _restrict_geometry_edges_to_box(self, gdf, box):
    gdf['geometry'] = gdf.geometry.intersection(box)
    return gdf


def _assign_labels_and_points_to_regions(
        self, gdf, cols_to_sort, col_label, col_point_label):
    # Firstly sort the regions so that selected regions appear first,
    # then both selected/not subsets are sorted by most northernly
    # first.
    gdf = gdf.sort_values(
        cols_to_sort, ascending=False
    )
    # Give each region a number.
    # The most-northernly selected region is 1,
    # and the least-northernly not-selected region is n.
    gdf[col_label] = np.arange(
        1, len(gdf) + 1).astype(str)
    # Get coordinates for where to plot each label:
    point_label = ([poly.representative_point() for
                    poly in gdf.geometry])
    gdf[col_point_label] = point_label
    return gdf


def _assign_labels_and_points_to_units(
        self,
        gdf_points_units,
        cols_to_sort,
        make_new_list=False
        ):
    """
    TO DO - write me.

    TO DO - add option to import this list from file
    or otherwise set custom labels.
    """
    if make_new_list:
        pass
    else:
        try:
            # If it exists, load in the existing labels DataFrame.
            df_unit_labels = self.df_unit_labels
            # Are there enough labels?
            gdf_points_units = pd.merge(
                gdf_points_units,
                df_unit_labels[['label', 'legend_order']],
                left_index=True, right_index=True, how='left')
            mask_missing = gdf_points_units['label'].isna()
            if mask_missing.any():
                # Missing some labels, so make more.
                used_labels = gdf_points_units['label'][~mask_missing].values
                units_without_labels = gdf_points_units.index[mask_missing].values
                # Remove the "labels" columns again.
                gdf_points_units = gdf_points_units.drop(
                    ['label', 'legend_order'], axis='columns')
            else:
                # Just sort the order of units:
                gdf_points_units = gdf_points_units.sort_values(
                    'legend_order')
                return gdf_points_units
        except AttributeError:
            # Make new labels instead.
            make_new_list = True

    if make_new_list:
        df_unit_labels = gdf_points_units[cols_to_sort].copy()
        df_unit_labels = df_unit_labels.sort_values(
            cols_to_sort, ascending=False
        )
        df_unit_labels['label'] = pd.NA
        df_unit_labels['legend_order'] = pd.NA
        used_labels = []
        units_without_labels = df_unit_labels.index.values

    new_labels = []
    # Do we need any extra labels?
    # Add a label letter for each unit.
    # List ['A', 'B', 'C', ..., 'Z']:
    new_labels = list(string.ascii_uppercase)
    # Remove anything already used:
    new_labels = [n for n in new_labels if n not in used_labels]

    if len(new_labels) < len(units_without_labels):
        # Add more letters at the end starting with
        # ['AA', 'AB', 'AC', ... 'AZ'].
        i = 0
        str_labels_orig = list(string.ascii_uppercase)
        while len(new_labels) < len(units_without_labels):
            str_labels2 = [f'{str_labels_orig[i]}{s}' for s in str_labels_orig]
            new_labels += str_labels2
            # Remove anything already used:
            new_labels = [n for n in new_labels if n not in used_labels]
            i += 1
    else:
        pass

    # Make a DataFrame of the new labels.
    new_labels = new_labels[:len(units_without_labels)]
    if df_unit_labels['legend_order'].isna().all():
        legend_order = np.arange(len(new_labels), dtype=int)
    else:
        start = 1 + df_unit_labels['legend_order'].max()
        legend_order = np.arange(start, start + len(new_labels), dtype=int)
    df_new_labels = pd.DataFrame(
        np.array([units_without_labels, new_labels, legend_order]).T,
        columns=['postcode', 'label', 'legend_order'])
    df_new_labels = df_new_labels.set_index('postcode')

    # Merge in these new labels:
    df_unit_labels = df_unit_labels.combine_first(df_new_labels)

    # Merge in to the starting DataFrame:
    gdf_units = pd.merge(
        gdf_points_units, df_unit_labels[['label', 'legend_order']],
        left_index=True, right_index=True, how='left')
    # Sort the order of units so the legend is alphabetical:
    gdf_units = gdf_units.sort_values('legend_order')
    self.df_unit_labels = df_unit_labels
    return gdf_units


def _combine_lsoa_into_catchment_shapes(self, gdf, col_to_dissolve):
    """
    # Combine LSOA geometry - from separate polygon per LSOA to one
    # big polygon for all LSOAs in catchment area.
    """
    # Assuming that index contains stuff to be dissolved,
    # it doesn't make sense to keep it afterwards.
    # Make it a normal column so it can be neglected.
    gdf = gdf.reset_index()
    gdf = gdf[[col_to_dissolve, 'geometry']].dissolve(by=col_to_dissolve)
    return gdf


def find_catchment_info_regions_and_units(
        self, df_catchment, df_units_regions
        ):
    """
    """
    # Find list of regions containing LSOA caught by selected units.
    regions_containing_lsoa = sorted(list(set(
        df_catchment['region_code'])))

    # Find list of units catching any LSOA in selected regions:
    units_catching_lsoa = sorted(list(set(
        df_catchment['unit_postcode'])))

    # Limit the units data:
    mask = df_units_regions['Postcode'].isin(units_catching_lsoa)
    df_units_regions = df_units_regions[mask]
    # Find list of regions containing these units:
    regions_containing_units_catching_lsoa = (
        df_units_regions['region_code'].tolist())

    to_return = (
        regions_containing_lsoa,
        units_catching_lsoa,
        regions_containing_units_catching_lsoa
    )
    return to_return