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


class Map(object):
    """
    Combine files from multiple runs of the pathway.

    class Combine():

    TO DO - write me
    """
    def __init__(self, *initial_data, **kwargs):

        # Overwrite default values
        # (can take named arguments or a dictionary)
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])

        for key in kwargs:
            setattr(self, key, kwargs[key])

        # If no setup was given, create one now:
        try:
            self.setup
        except AttributeError:
            self.setup = Setup()

    # ##########################
    # ##### FILE SELECTION #####
    # ##########################
    def set_data_dir(self, dir_data):
        """
        On changing dir, wipe the loaded data and reload from this dir.
        """
        if dir_data is None:
            # Use combined data files.
            self.dir_data = self.setup.dir_output_combined
            self.data_type = 'combined'
        else:
            # Use files for the selected scenario only.
            for d in self.setup.list_dir_output:
                end = os.path.split(d)[-1]
                if end == dir_data:
                    self.dir_data = d
            self.data_type = 'single'

        self._delete_loaded_data()

    def _delete_loaded_data(self):
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
        """
        # Setup for combined files:
        dicts_combo = {
            'df_regions': {
                'file': self.setup.file_combined_selected_regions,
                'header': [0, 1],
                'index_col': 1,
                },
            'df_units': {
                'file': self.setup.file_combined_selected_stroke_units,
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
                'index_col': 1,
                },
            'df_units': {
                'file': self.setup.file_selected_stroke_units,
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
                pass
                # if data_dict['file'] is None:
                #     # Hit this on purpose. Carry on.
                #     pass
                # else:
                #     # Don't raise an error.
                #     # Can reach this condition when the file doesn't
                #     # exist yet and isn't needed yet. Just print info.
                #     # print(f'Cannot import {label} from {data_dict["file"]}')
                #     # raise FileNotFoundError(
                #     #     f'Cannot import {label} from {data_dict["file"]}'
                #     #     ) from None

    def process_data(self):
        """
        Load it in mmkay  TO DO - write me
        """
        def _check_prereqs_and_load(prereqs, func):
            data_loaded = [hasattr(self, df) for df in prereqs]
            if all(data_loaded):
                func()
            else:
                # Something is missing so can't load the data.
                pass

        func = self._load_geometry_regions
        prereqs = ['df_regions']
        _check_prereqs_and_load(prereqs, func)

        func = self._load_geometry_stroke_units
        prereqs = ['df_regions', 'df_units']
        _check_prereqs_and_load(prereqs, func)

        func = self._load_geometry_transfer_units
        prereqs = ['df_regions', 'df_units', 'df_transfer']
        _check_prereqs_and_load(prereqs, func)

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
            print('Missing some information: ', missing_attrs)
            print('Try reloading: self.load_run_data() and self.process_data()')
            return
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
            # 'ICB22NM': self.setup.file_geojson_icb,
            'LHB20NM': self.setup.file_geojson_lhb,
        }

        # Import region file:
        dir_input = self.setup.dir_data_geojson
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
    def _load_geometry_regions(self):
        """
        Create GeoDataFrames of new geometry and existing DataFrames.
        """
        # ----- Gather data -----
        # Selected regions names and usage.
        # ['{region type}', 'contains_selected_unit', 'contains_selected_lsoa']
        df_regions = self.df_regions
        # Index column: 'region'.
        # Expected column MultiIndex levels:
        #   - combined: ['scenario', 'property']
        #   - separate: ['{unnamed level}']

        # All region polygons:
        gdf_boundaries_regions_e = self.import_geojson('SICBL22NM')
        gdf_boundaries_regions_w = self.import_geojson('LHB20NM')
        # Combine:
        gdf_boundaries_regions = pd.concat(
            (gdf_boundaries_regions_e, gdf_boundaries_regions_w),
            axis='rows'
        )
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
        # Selected stroke units names, coordinates, and services.
        df_units = self.df_units
        # Index column: Postcode.
        # Expected column MultiIndex levels:
        #   - combined: ['scenario', 'property']
        #   - separate: ['{unnamed level}']

        if self.data_type == 'combined':
            x_col = ('any', 'Easting')
            y_col = ('any', 'Northing')
            coords_col = ('any', 'geometry')
        else:
            x_col = 'Easting'
            y_col = 'Northing'
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

    def _load_geometry_transfer_units(self):
        """
        Create GeoDataFrames of new geometry and existing DataFrames.
        """
        # Selected stroke units names, coordinates, and services.
        df_transfer = self.df_transfer
        # Index column: ['Postcode', 'name_nearest_mt']
        # Expected column MultiIndex levels:
        #   - combined: ['scenario', 'property']
        #   - separate: ['{unnamed level}']

        if self.data_type == 'combined':
            # From the loaded file:
            x_col = ('any', 'Easting')
            y_col = ('any', 'Northing')
            x_col_mt = ('any', 'Easting_mt')
            y_col_mt = ('any', 'Northing_mt')
            # To be created here:
            col_unit = ('any', 'unit_coords')
            col_tran = ('any', 'transfer_coords')
            col_line_coords = ('any', 'line_coords')
            col_line_geometry = ('any', 'geometry')
        else:
            # From the loaded file:
            x_col = 'Easting'
            y_col = 'Northing'
            x_col_mt = 'Easting_mt'
            y_col_mt = 'Northing_mt'
            # To be created here:
            col_unit = 'unit_coords'
            col_tran = 'transfer_coords'
            col_line_coords = 'line_coords'
            col_line_geometry = 'geometry'

        # Convert to geometry (line):

        # Make a column of coordinates [x, y]:
        xy = df_transfer[[x_col, y_col]]
        df_transfer[col_unit] = xy.values.tolist()

        xy_mt = df_transfer[[x_col_mt, y_col_mt]]
        df_transfer[col_tran] = xy_mt.values.tolist()

        gdf_transfer = self.create_lines_from_coords(
            df_transfer,
            [col_unit, col_tran],
            col_line_coords,
            col_line_geometry
            )

        self.gdf_lines_transfer = gdf_transfer

    def _load_geometry_lsoa(self):
        """
        Create GeoDataFrames of new geometry and existing DataFrames.
        """
        # ----- Gather data -----
        # Selected LSOA names, codes.
        # ['LSOA11NM', 'LSOA11CD', 'postcode_nearest', 'time_nearest', 'Use']
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
        df[col_geom] = [LineString(coords) for coords in df[col_coord]]

        # Convert to GeoDataFrame:
        gdf = geopandas.GeoDataFrame(df, geometry=col_geom)  #, crs="EPSG:4326"
        # if isinstance(col_geom, tuple):
        #     gdf['geometry']
        # TO DO - implement CRS explicitly ---------------------------------------------
        return gdf

    def _find_use_column_for_transfer_lines(self, gdf_transfer, df_units):
        # Work out which lines should be used.
        index_cols = gdf_transfer.index.names
        gdf_transfer = gdf_transfer.reset_index().copy()
        # Set 'Use' to 1 when either the start or end unit
        # is in 'selected'.
        gdf_transfer['Use'] = 0
        df_units_rs = df_units.copy().reset_index()
        # Is start unit in selected?
        gdf_transfer = pd.merge(
            gdf_transfer,
            df_units_rs[['Postcode', 'selected']],
            left_on='Postcode', right_on='Postcode', how='left'
        )
        # Is end unit in selected?
        gdf_transfer = pd.merge(
            gdf_transfer,
            df_units_rs[['Postcode', 'selected']],
            left_on='name_nearest_mt', right_on='Postcode', how='left',
            suffixes=(None, '_mt')
        )
        gdf_transfer['Use'][(
            (gdf_transfer['selected'] == 1) |
            (gdf_transfer['selected_mt'] == 1)
            )] = 1
        gdf_transfer = gdf_transfer.set_index(index_cols)
        return gdf_transfer

    def create_combo_cols(self, gdf, scenario):
        """
        TO DO - write me
        When dataframe doesn't have the diff_this_minus_that columns,
        use this function to create that data and prevent KeyError later.
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
            regions_selected,
            col_region,
            col_unit_selected,
            cols_to_sort,
            col_unit_label
            ):
        # Which stroke units are in the selected regions?
        mask = gdf_points_units[col_region].isin(regions_selected)
        gdf_points_units[col_unit_selected] = 0
        gdf_points_units.loc[mask, col_unit_selected] = 1
        # Add a label letter for each unit.
        gdf_points_units = gdf_points_units.sort_values(
            cols_to_sort, ascending=False
        )
        # List ['A', 'B', 'C', ..., 'Z']:
        str_labels = list(string.ascii_uppercase)
        if len(str_labels) < len(gdf_points_units):
            # Add more letters at the end starting with
            # ['AA', 'AB', 'AC', ... 'AZ'].
            i = 0
            str_labels_orig = list(string.ascii_uppercase)
            while len(str_labels) < len(gdf_points_units):
                str_labels2 = [f'{str_labels[i]}{s}' for s in str_labels_orig]
                str_labels += str_labels2
                i += 1
        else:
            pass
        gdf_points_units[col_unit_label] = str_labels[:len(gdf_points_units)]
        return gdf_points_units

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

    # #########################
    # ##### PLOT WRAPPERS #####
    # #########################
    def plot_map_selected_regions(
            self,
            scenario: str,
            save=True,
            show=False
            ):
        """
        Wrangle data and plot a map of selected units.
        """
        self.load_run_data()
        self.process_data()

        map_args, map_kwargs = self._setup_plot_map_selected_regions(
            scenario,
            save
            )
        self._plt_plot_map_selected_regions(
            *map_args,
            **map_kwargs,
            save=save,
            show=show
        )

    def plot_map_selected_units(
            self,
            scenario: str,
            save=True,
            show=False
            ):
        """
        Wrangle data and plot a map of selected units.
        """
        self.load_run_data()
        self.process_data()

        map_args, map_kwargs = self._setup_plot_map_selected_units(
            scenario,
            save
            )
        self._plt_plot_map_selected_units(
            *map_args,
            **map_kwargs,
            save=save,
            show=show
        )

    def plot_map_catchment(
            self,
            scenario: str,
            catchment_type='',
            save=True,
            show=False
            ):
        """
        Wrangle data and plot a map of selected unit catchments.
        """
        self.load_run_data()
        self.process_data()

        map_args, map_kwargs = self._setup_plot_map_catchment(
            scenario,
            catchment_type=catchment_type,
            save=save
            )
        self._plt_plot_map_catchment(
            *map_args,
            **map_kwargs,
            # catchment_type=catchment_type,
            save=save,
            show=show
        )

    def plot_map_outcome(
            self,
            scenario: str,
            outcome: str,
            catchment_type='',
            save=True,
            show=False,
            region=None,
            boundary_kwargs={},
            ):
        """
        Wrangle data and plot a map of LSOA outcomes.
        """
        self.load_run_data()
        self.process_data()

        map_args, map_kwargs = self._setup_plot_map_outcome(
            scenario,
            outcome,
            boundary_kwargs=boundary_kwargs,
            catchment_type=catchment_type,
            save=save
            )
        self._plt_plot_map_outcome(
            *map_args,
            **map_kwargs,
            title=f'{scenario}\n{outcome}',
            save=save,
            show=show
        )

    # ###########################
    # ##### SETUP FOR PLOTS #####
    # ###########################
    def _setup_plot_map_selected_regions(
            self,
            scenario: str,
            save=True
            ):
        # Check whether everything we need exists.
        prereqs = ['gdf_boundaries_regions', 'gdf_points_units']
        self._check_prereqs_exist(prereqs)

        if self.data_type == 'combined':
            # Remove excess scenario data:
            try:
                c = ['any', scenario]
                gdf_boundaries_regions = self.gdf_boundaries_regions[c].copy()
                gdf_points_units = self.gdf_points_units[c].copy()
            except KeyError:
                # The scenario isn't in the Data.
                err = f'{scenario} scenario is missing from combined data.'
                raise KeyError(err) from None

            # Remove the 'scenario' column heading:
            gdf_boundaries_regions = self._remove_excess_heading_from_gdf(
                gdf_boundaries_regions, 0, 'geometry')
            gdf_points_units = self._remove_excess_heading_from_gdf(
                gdf_points_units, 0, 'geometry')
        else:
            gdf_boundaries_regions = self.gdf_boundaries_regions.copy()
            gdf_points_units = self.gdf_points_units.copy()

        box, map_extent = self.get_selected_area_extent(
            gdf_boundaries_regions[gdf_boundaries_regions['selected'] == 1])
        gdf_boundaries_regions = self._keep_only_geometry_in_box(
            gdf_boundaries_regions, box)
        gdf_boundaries_regions = self._restrict_geometry_edges_to_box(
            gdf_boundaries_regions, box)
        gdf_points_units = self._keep_only_geometry_in_box(
            gdf_points_units, box)

        # Create labels *after* choosing the map
        # extent and restricting the regions to the edges of the box.
        # Otherwise labels could appear outside the plot and
        # all the good labels would be assigned to places not shown.
        gdf_boundaries_regions = self._assign_labels_and_points_to_regions(
            gdf_boundaries_regions,
            ['selected', 'BNG_N'], 'label', 'point_label')

        regions_selected = gdf_boundaries_regions['region'][
            gdf_boundaries_regions['selected'] == 1]
        gdf_points_units = self._assign_labels_and_points_to_units(
            gdf_points_units,
            regions_selected,
            'region',
            'selected',
            ['selected', 'Northing'],
            'label'
            )

        # Reduce the DataFrames to just the needed parts:
        gdf_boundaries_regions = gdf_boundaries_regions[[
            'geometry',                             # shapes
            'selected',                             # line type selection
            'label',                                # label annotation
            'point_label',                          # label position
            'region',                               # legend label
            ]]
        gdf_points_units = gdf_points_units[[
            'geometry',                             # locations
            'use_ivt', 'use_mt', 'use_msu',         # point selection
            'Hospital_name',                        # labels
            'label',                                # label annotation
            'selected'                              # ordering labels
            ]]

        # Create file name:
        if save:
            file_name = f'map_selected_regions_{scenario}.jpg'
            path_to_file = os.path.join(self.dir_data, file_name)
        else:
            path_to_file = None

        map_args = (
            gdf_boundaries_regions,
            gdf_points_units
            )
        map_kwargs = dict(
            map_extent=map_extent,
            path_to_file=path_to_file
            )
        return map_args, map_kwargs

    def _setup_plot_map_selected_units(
            self,
            scenario,
            save=True
            ):
        # Check whether everything we need exists.
        prereqs = ['gdf_boundaries_regions', 'gdf_points_units',
                   'gdf_lines_transfer']
        self._check_prereqs_exist(prereqs)

        if self.data_type == 'combined':
            # Remove excess scenario data:
            try:
                c = ['any', scenario]
                gdf_boundaries_regions = self.gdf_boundaries_regions[c].copy()
                gdf_points_units = self.gdf_points_units[c].copy()
                gdf_lines_transfer = self.gdf_lines_transfer[c].copy()
            except KeyError:
                # The scenario isn't in the Data.
                err = f'{scenario} scenario is missing from combined data.'
                raise KeyError(err) from None

            # Remove the 'scenario' column heading:
            gdf_boundaries_regions = self._remove_excess_heading_from_gdf(
                gdf_boundaries_regions, 0, 'geometry')
            gdf_points_units = self._remove_excess_heading_from_gdf(
                gdf_points_units, 0, 'geometry')
            gdf_lines_transfer = self._remove_excess_heading_from_gdf(
                gdf_lines_transfer, 0, 'geometry')
        else:
            gdf_boundaries_regions = self.gdf_boundaries_regions.copy()
            gdf_points_units = self.gdf_points_units.copy()
            gdf_lines_transfer = self.gdf_lines_transfer.copy()

        box, map_extent = self.get_selected_area_extent(
            gdf_boundaries_regions[gdf_boundaries_regions['selected'] == 1])
        gdf_boundaries_regions = self._keep_only_geometry_in_box(
            gdf_boundaries_regions, box)
        gdf_boundaries_regions = self._restrict_geometry_edges_to_box(
            gdf_boundaries_regions, box)
        gdf_points_units = self._keep_only_geometry_in_box(
            gdf_points_units, box)

        # Create labels *after* choosing the map
        # extent and restricting the regions to the edges of the box.
        # Otherwise labels could appear outside the plot and
        # all the good labels would be assigned to places not shown.
        regions_selected = gdf_boundaries_regions['region'][
            gdf_boundaries_regions['selected'] == 1]
        gdf_points_units = self._assign_labels_and_points_to_units(
            gdf_points_units,
            regions_selected,
            'region',
            'selected',
            ['selected', 'Northing'],
            'label'
            )

        gdf_lines_transfer = self._find_use_column_for_transfer_lines(
            gdf_lines_transfer, gdf_points_units)

        # Reduce the DataFrames to just the needed parts:
        gdf_boundaries_regions = gdf_boundaries_regions[[
            'geometry',                             # shapes
            'selected',                             # line type selection
            ]]
        gdf_points_units = gdf_points_units[[
            'geometry',                             # locations
            'use_ivt', 'use_mt', 'use_msu',         # point selection
            'Hospital_name',                        # labels
            'label',                                # label annotation
            'selected'                              # label kwargs
            ]]
        gdf_lines_transfer = gdf_lines_transfer[[
            'geometry',                             # line end points
            'Use'                                   # line selection
            ]]

        # Create file name:
        if save:
            file_name = f'map_selected_units_{scenario}.jpg'
            path_to_file = os.path.join(self.dir_data, file_name)
        else:
            path_to_file = None

        map_args = (
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer
        )
        map_kwargs = dict(
            path_to_file=path_to_file,
            map_extent=map_extent
            )
        return map_args, map_kwargs

    def _setup_plot_map_catchment(
            self,
            scenario: str,
            catchment_type: str = '',
            boundary_kwargs={},
            save=True
            ):
        """

        """
        # Check whether everything we need exists.
        prereqs = ['gdf_boundaries_regions', 'gdf_points_units',
                   'gdf_boundaries_lsoa', 'gdf_lines_transfer']
        self._check_prereqs_exist(prereqs)

        if self.data_type == 'combined':
            # Remove excess scenario data:
            try:
                c = ['any', scenario]
                gdf_boundaries_lsoa = self.gdf_boundaries_lsoa[c]
                gdf_boundaries_regions = self.gdf_boundaries_regions[c]
                gdf_points_units = self.gdf_points_units[c]
                gdf_lines_transfer = self.gdf_lines_transfer[c]
            except KeyError:
                # The scenario isn't in the Data.
                err = f'{scenario} scenario is missing from combined data.'
                raise KeyError(err) from None

            # Remove the 'scenario' column heading:
            gdf_boundaries_lsoa = self._remove_excess_heading_from_gdf(
                gdf_boundaries_lsoa, 0, ('geometry', ''))
            gdf_boundaries_regions = self._remove_excess_heading_from_gdf(
                gdf_boundaries_regions, 0, 'geometry')
            gdf_points_units = self._remove_excess_heading_from_gdf(
                gdf_points_units, 0, 'geometry')
            gdf_lines_transfer = self._remove_excess_heading_from_gdf(
                gdf_lines_transfer, 0, 'geometry')
            # Remove the 'subtype' column heading:
            gdf_boundaries_lsoa = self._remove_excess_heading_from_gdf(
                gdf_boundaries_lsoa, 1, 'geometry')
        else:
            gdf_boundaries_lsoa = self.gdf_boundaries_lsoa.copy()
            gdf_boundaries_regions = self.gdf_boundaries_regions.copy()
            gdf_points_units = self.gdf_points_units.copy()
            gdf_lines_transfer = self.gdf_lines_transfer.copy()

            # Remove the 'subtype' column heading:
            gdf_boundaries_lsoa = self._remove_excess_heading_from_gdf(
                gdf_boundaries_lsoa, 1, 'geometry')

        # Combine LSOA geometry - from separate polygon per LSOA to one
        # big polygon for all LSOAs in catchment area.
        #  TO DO - might not always be called _ivt in following column:
        # TO DO - save this output file as .geojson for future use.
        gdf_boundaries_lsoa = self._combine_lsoa_into_catchment_shapes(
            gdf_boundaries_lsoa, 'postcode_nearest_ivt')

        if catchment_type == 'island':
            # TO DO - currently this means that stroke units get different labels between 
            # this map and the other stroke unit selection map. Fix it! ------------------------

            # Only keep selected regions.
            mask = (gdf_boundaries_regions['selected'] == 1)
            gdf_boundaries_regions = gdf_boundaries_regions[mask]
            # Which stroke units are contained in this bounding box?
            mask = (gdf_points_units['selected'] == 1)
            gdf_points_units = gdf_points_units[mask]

            # Take map extent from the region geometry
            # *after* removing unwanted regions.
            box, map_extent = self.get_selected_area_extent(
                gdf_boundaries_regions)
        else:
            # Take map extent from the combined LSOA and region geometry.
            gdf_regions_reduced = gdf_boundaries_regions.copy()[
                gdf_boundaries_regions['selected'] == 1
                ].reset_index()['geometry']
            gdf_lsoa_reduced = gdf_boundaries_lsoa.copy(
                ).reset_index()['geometry']
            gdf_combo = pd.concat(
                (gdf_regions_reduced, gdf_lsoa_reduced), axis='rows')

            # TO DO - currently this means that stroke units get different labels between 
            # this map and the other stroke unit selection map. Fix it! ------------------------

            box, map_extent = self.get_selected_area_extent(gdf_combo)
            gdf_boundaries_regions = self._keep_only_geometry_in_box(
                gdf_boundaries_regions, box)
            gdf_points_units = self._keep_only_geometry_in_box(
                gdf_points_units, box)

        # Restrict polygon geometry to the edges of the box.
        gdf_boundaries_regions = self._restrict_geometry_edges_to_box(
            gdf_boundaries_regions, box)

        # Create labels *after* choosing the map
        # extent and restricting the regions to the edges of the box.
        # Otherwise labels could appear outside the plot and
        # all the good labels would be assigned to places not shown.
        regions_selected = gdf_boundaries_regions['region'][
            gdf_boundaries_regions['selected'] == 1]
        gdf_points_units = self._assign_labels_and_points_to_units(
            gdf_points_units,
            regions_selected,
            'region',
            'selected',
            ['selected', 'Northing'],
            'label'
            )

        gdf_lines_transfer = self._find_use_column_for_transfer_lines(
            gdf_lines_transfer, gdf_points_units)

        # TO DO - make gdf_boundaries_regions contains_selected_lsoa column.

        # Reduce the DataFrames to just the needed parts:
        gdf_boundaries_lsoa = gdf_boundaries_lsoa[[
            'geometry',                             # shapes
            ]]
        gdf_boundaries_regions = gdf_boundaries_regions[[
            'geometry',                             # shapes
            'selected',                             # line type selection
            # 'contains_selected_lsoa',               # line type selection
            ]]
        gdf_points_units = gdf_points_units[[
            'geometry',                             # locations
            'use_ivt', 'use_mt', 'use_msu',  # 'Use',  # point selection
            'Hospital_name',                        # labels
            'label',                                # label annotation
            'selected'                              # label kwargs
            ]]
        gdf_lines_transfer = gdf_lines_transfer[[
            'geometry',                             # line end points
            'Use'                                   # line selection
        ]]

        lsoa_boundary_kwargs = {
            'cmap': 'Blues',
            'edgecolor': 'face'
        }
        # Update this with anything from the input dict:
        lsoa_boundary_kwargs = lsoa_boundary_kwargs | boundary_kwargs

        if save:
            file_name = f'map_catchment_{scenario}'
            if len(catchment_type) > 0:
                file_name += f'_{catchment_type}'
            file_name += '.jpg'

            path_to_file = os.path.join(self.dir_data, file_name)
        else:
            path_to_file = None

        map_args = (
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer
            )
        map_kwargs = dict(
            lsoa_boundary_kwargs=lsoa_boundary_kwargs,
            map_extent=map_extent,
            path_to_file=path_to_file
        )
        return map_args, map_kwargs

    def _setup_plot_map_outcome(
            self,
            scenario: str,
            outcome: str,
            boundary_kwargs={},
            catchment_type='',
            save=True
            ):
        """

        """
        # Check whether everything we need exists.
        prereqs = ['gdf_boundaries_regions', 'gdf_points_units',
                   'gdf_boundaries_lsoa']
        self._check_prereqs_exist(prereqs)

        gdf_boundaries_lsoa = self.gdf_boundaries_lsoa.copy()
        gdf_boundaries_regions = self.gdf_boundaries_regions.copy()
        gdf_points_units = self.gdf_points_units.copy()

        if self.data_type == 'combined':
            # Find shared outcome limits.
            # Take only scenarios containing 'diff':
            mask = gdf_boundaries_lsoa.columns.get_level_values(
                    0).str.contains('diff')
            if outcome.startswith('diff'):
                pass
            else:
                # Take the opposite condition, take only scenarios
                # not containing 'diff'.
                mask = ~mask

            mask = (
                mask &
                (gdf_boundaries_lsoa.columns.get_level_values(2) == 'mean') &
                (gdf_boundaries_lsoa.columns.get_level_values(1) == outcome)
            )
            all_mean_vals = gdf_boundaries_lsoa.iloc[:, mask]
            vlim_abs = all_mean_vals.abs().max().values[0]
            vmax = all_mean_vals.max().values[0]
            vmin = all_mean_vals.min().values[0]

            if 'diff' in scenario:
                # Add any other columns that these expect.
                gdf_boundaries_regions = self.create_combo_cols(
                    gdf_boundaries_regions, scenario)
                gdf_points_units = self.create_combo_cols(
                    gdf_points_units, scenario)

            try:
                # Remove excess scenario data:
                s = ['any', scenario]
                gdf_boundaries_lsoa = gdf_boundaries_lsoa[s]
                gdf_boundaries_regions = gdf_boundaries_regions[s]
                gdf_points_units = gdf_points_units[s]
            except KeyError:
                # The scenario isn't in the Data.
                err = f'{scenario} scenario is missing from combined data.'
                raise KeyError(err) from None

            # Remove the "std" columns.
            mask = (gdf_boundaries_lsoa.columns.get_level_values(2) == 'std')
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.drop(
                gdf_boundaries_lsoa.iloc[:, mask].columns,
                axis='columns'
            )

            # Remove the 'scenario' column heading:
            gdf_boundaries_lsoa = self._remove_excess_heading_from_gdf(
                gdf_boundaries_lsoa, 0, ('geometry', ''))
            gdf_boundaries_regions = self._remove_excess_heading_from_gdf(
                gdf_boundaries_regions, 0, 'geometry')
            gdf_points_units = self._remove_excess_heading_from_gdf(
                gdf_points_units, 0, 'geometry')
            # Remove the 'subtype' column heading:
            gdf_boundaries_lsoa = self._remove_excess_heading_from_gdf(
                gdf_boundaries_lsoa, 1, 'geometry')
        else:
            # Find the colour limits.
            mask = (
                (gdf_boundaries_lsoa.columns.get_level_values(1) == 'mean') &
                (gdf_boundaries_lsoa.columns.get_level_values(0) == outcome)
            )
            mean_vals = gdf_boundaries_lsoa.iloc[:, mask]
            vlim_abs = mean_vals.abs().max().values[0]
            vmax = mean_vals.max().values[0]
            vmin = mean_vals.min().values[0]

            # Remove the "std" columns.
            mask = (gdf_boundaries_lsoa.columns.get_level_values(1) == 'std')
            gdf_boundaries_lsoa = gdf_boundaries_lsoa.drop(
                gdf_boundaries_lsoa.iloc[:, mask].columns,
                axis='columns'
            )

            # Remove the 'subtype' column heading:
            gdf_boundaries_lsoa = self._remove_excess_heading_from_gdf(
                gdf_boundaries_lsoa, 1, 'geometry')

        if catchment_type == 'island':
            # TO DO - currently this means that stroke units get different labels between 
            # this map and the other stroke unit selection map. Fix it! ------------------------

            # Only keep selected regions.
            mask = (gdf_boundaries_regions['selected'] == 1)
            gdf_boundaries_regions = gdf_boundaries_regions[mask]

            # Take map extent from the region geometry
            # *after* removing unwanted regions.
            box, map_extent = self.get_selected_area_extent(
                gdf_boundaries_regions,
                leeway=20000,
                )
        else:
            # Take map extent from the combined LSOA and region geometry.
            gdf_regions_reduced = gdf_boundaries_regions.copy()[
                gdf_boundaries_regions['selected'] == 1
                ].reset_index()['geometry']
            gdf_lsoa_reduced = gdf_boundaries_lsoa.copy(
                ).reset_index()['geometry']
            gdf_combo = pd.concat(
                (gdf_regions_reduced, gdf_lsoa_reduced), axis='rows')

            # TO DO - currently this means that stroke units get different labels between 
            # this map and the other stroke unit selection map. Fix it! ------------------------

            box, map_extent = self.get_selected_area_extent(
                gdf_combo,
                leeway=20000,
                )
            gdf_boundaries_regions = self._keep_only_geometry_in_box(
                gdf_boundaries_regions, box)

        # Only keep selected stroke units.
        mask = (gdf_points_units['selected'] == 1)
        gdf_points_units = gdf_points_units[mask]

        gdf_boundaries_regions = self._restrict_geometry_edges_to_box(
            gdf_boundaries_regions, box)

        # Create labels *after* choosing the map
        # extent and restricting the regions to the edges of the box.
        # Otherwise labels could appear outside the plot and
        # all the good labels would be assigned to places not shown.
        regions_selected = gdf_boundaries_regions['region'][
            gdf_boundaries_regions['selected'] == 1]
        gdf_points_units = self._assign_labels_and_points_to_units(
            gdf_points_units,
            regions_selected,
            'region',
            'selected',
            ['selected', 'Northing'],
            'label'
            )

        # TO DO - make gdf_boundaries_regions contains_selected_lsoa column.

        # Reduce the DataFrames to just the needed parts:
        gdf_boundaries_lsoa = gdf_boundaries_lsoa[[
            'geometry',                             # shapes
            outcome                                 # colours
            ]]
        gdf_boundaries_regions = gdf_boundaries_regions[[
            'geometry',                             # shapes
            'selected',                             # line type selection
            # 'contains_selected_lsoa',               # line type selection
            ]]
        gdf_points_units = gdf_points_units[[
            'geometry',                             # locations
            'use_ivt', 'use_mt', 'use_msu',# 'Use',  # point selection
            'Hospital_name',                        # labels
            'label',                                # label annotation
            'selected'                       # label kwargs
            ]]

        lsoa_boundary_kwargs = {
            'column': outcome,
            'edgecolor': 'face',
            # Adjust size of colourmap key, and add label
            'legend_kwds': {
                'shrink': 0.5,
                'label': outcome
                },
            # Set to display legend
            'legend': True,
            }

        cbar_diff = True if outcome.startswith('diff') else False
        if cbar_diff:
            lsoa_boundary_kwargs['cmap'] = 'seismic'
            lsoa_boundary_kwargs['vmin'] = -vlim_abs
            lsoa_boundary_kwargs['vmax'] = vlim_abs
        else:
            cmap = 'plasma'
            if outcome == 'mRS shift':
                # Reverse the colourmap because lower values
                # are better for this outcome.
                cmap += '_r'
            lsoa_boundary_kwargs['cmap'] = cmap
            lsoa_boundary_kwargs['vmin'] = vmin
            lsoa_boundary_kwargs['vmax'] = vmax
        # Update this with anything from the input dict:
        lsoa_boundary_kwargs = lsoa_boundary_kwargs | boundary_kwargs

        if save:
            file_name = f'map_{outcome}_{scenario}.jpg'
            path_to_file = os.path.join(self.dir_data, file_name)
        else:
            path_to_file = None

        map_args = (
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units
        )
        map_kwargs = dict(
            lsoa_boundary_kwargs=lsoa_boundary_kwargs,
            path_to_file=path_to_file
        )
        return map_args, map_kwargs

    # #######################
    # ##### PYPLOT MAPS #####
    # #######################
    def _plt_plot_map_selected_regions(
            self,
            gdf_boundaries_regions,
            gdf_points_units,
            map_extent=[],
            path_to_file='',
            save=True,
            show=False
            ):
        fig, ax = plt.subplots(figsize=(8, 5))

        ax, extra_artists = maps.plot_map_selected_regions(
            gdf_boundaries_regions,
            gdf_points_units,
            ax=ax,
            map_extent=map_extent
        )

        if save:
            # Return extra artists so that bbox_inches='tight' line
            # in savefig() doesn't cut off the legends.
            # Adding legends with ax.add_artist() means that the
            # bbox_inches='tight' line ignores them.
            plt.savefig(
                path_to_file,
                bbox_extra_artists=extra_artists,
                dpi=300, bbox_inches='tight'
                )
        else:
            pass
        if show:
            # Add dummy axis to the sides so that
            # extra_artists are not cut off when plt.show() crops.
            fig = maps.plot_dummy_axis(fig, ax, extra_artists[1], side='left')
            fig = maps.plot_dummy_axis(fig, ax, extra_artists[0], side='right')
            plt.show()
        else:
            plt.close()

    def _plt_plot_map_selected_units(
            self,
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer,
            map_extent=[],
            path_to_file='',
            save=True,
            show=False
            ):
        fig, ax = plt.subplots(figsize=(6, 5))

        ax, extra_artists = maps.plot_map_selected_units(
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer,
            ax=ax,
            map_extent=map_extent,
        )

        if save:
            plt.savefig(
                path_to_file,
                bbox_extra_artists=extra_artists,
                dpi=300, bbox_inches='tight'
                )
        else:
            pass
        if show:
            # Add dummy axis to the sides so that
            # extra_artists are not cut off when plt.show() crops.
            fig = maps.plot_dummy_axis(fig, ax, extra_artists[0], side='left')
            plt.show()
        else:
            plt.close()

    def _plt_plot_map_catchment(
            self,
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer,
            title='',
            lsoa_boundary_kwargs={},
            map_extent=[],
            # catchment_type='',
            save=True,
            show=False,
            path_to_file=''
            ):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.set_title(title)

        ax, extra_artists = maps.plot_map_catchment(
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            gdf_lines_transfer,
            ax=ax,
            map_extent=map_extent,
            boundary_kwargs=lsoa_boundary_kwargs,
            # catchment_type=catchment_type
        )

        if save:
            plt.savefig(
                path_to_file,
                bbox_extra_artists=extra_artists,
                dpi=300, bbox_inches='tight')
        else:
            pass
        if show:
            # Add dummy axis to the sides so that
            # extra_artists are not cut off when plt.show() crops.
            fig = maps.plot_dummy_axis(fig, ax, extra_artists[0], side='left')
            plt.show()
        else:
            plt.close()

    def _plt_plot_map_outcome(
            self,
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            title='',
            lsoa_boundary_kwargs={},
            save=True,
            show=False,
            path_to_file=None
            ):
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.set_title(title)

        ax, extra_artists = maps.plot_map_outcome(
            gdf_boundaries_lsoa,
            gdf_boundaries_regions,
            gdf_points_units,
            ax=ax,
            boundary_kwargs=lsoa_boundary_kwargs
        )

        if save:
            plt.savefig(
                path_to_file,
                bbox_extra_artists=extra_artists,
                dpi=300, bbox_inches='tight')
        else:
            pass
        if show:
            # Add dummy axis to the sides so that
            # extra_artists are not cut off when plt.show() crops.
            fig = maps.plot_dummy_axis(fig, ax, extra_artists[0], side='left')
            plt.show()
        else:
            plt.close()
