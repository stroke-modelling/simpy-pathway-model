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


    # ----- Gather data -----
    # Selected LSOA names, codes.
    # ['lsoa', 'LSOA11CD', 'postcode_nearest', 'time_nearest', 'Use']
    # Index column: LSOA11NM.
    # Expected column MultiIndex levels:
    #   - combined: ['scenario', 'property']
    #   - separate: ['{unnamed level}']
"""
import numpy as np
import pandas as pd
import geopandas
import os

from shapely import LineString  # For creating line geometry.



# #####################
# ##### LOAD DATA #####
# #####################
def import_geojson(region_type: 'str'):
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
        'LSOA11NM': ''.join([
            'LSOA_(Dec_2011)_Boundaries_Super_Generalised_Clipped_(BSC)',
            '_EW_V3.geojson'
        ]),
        'SICBL22NM': 'SICBL_JUL_2022_EN_BUC_4104971945004813003.geojson',
        'LHB20NM': ''.join([
            'Local_Health_Boards_April_2020_WA_BGC_2022_',
            '94310626700012506.geojson'
        ]),
    }

    # TO DO - change to relative import.
    # # Relative import from package files:
    # path_to_file = files('map.data_geojson').joinpath(file_input)
    # Import region file:
    file_input = geojson_file_dict[region_type]
    path_to_file = os.path.join('data_geojson', file_input)
    gdf_boundaries = geopandas.read_file(path_to_file)

    if region_type == 'LSOA11NM':
        index_col = 'LSOA11CD'
        # Only keep these columns.
        geo_cols = ['LSOA11NM', 'BNG_E', 'BNG_N',
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


# ########################
# ##### PROCESS DATA #####
# ########################
def load_regions():
    """
    Load region data from file.
    """

    # Load and parse geometry data
    # TO DO - change to relative import.
    # # Relative import from package files:
    # path_to_file = files('map.data').joinpath('regions_ew.csv')
    path_to_file = os.path.join('data', 'regions_ew.csv')
    df_regions = pd.read_csv(path_to_file, index_col=[0, 1])

    # Add an extra column level:
    # Everything needs two levels: scenario, property.
    cols_df_regions = [
        df_regions.columns,                 # property
        ['any'] * len(df_regions.columns),  # scenario
    ]
    # New DataFrame with the extra column level:
    df_regions = pd.DataFrame(
        df_regions.values,
        index=df_regions.index,
        columns=cols_df_regions
    )
    # Index column: 'region'.
    # Expected column MultiIndex levels:
    #   - combined: ['scenario', 'property']
    #   - separate: ['{unnamed level}']

    return df_regions


def make_new_periphery_data(
        df_regions, df_units, df_lsoa):
    """
    Find units, regions that aren't selected but catch selected LSOA.
    """
    # List of scenarios included in the units and LSOA data:
    scenario_list = sorted(list(set(
        df_units.columns.get_level_values('scenario'))))
    scenario_list.remove('any')

    # Load region info for each LSOA:
    # TO DO - change to relative import.
    # # Relative import from package files:
    # path_to_file = files('map.data').joinpath('regions_lsoa_ew.csv')
    # Import region file:
    path_to_file = os.path.join('data', 'regions_lsoa_ew.csv')
    df_lsoa_regions = pd.read_csv(path_to_file, index_col=[0, 1])
    # Add an extra column level:
    # Everything needs two levels: scenario, property.
    cols_df_lsoa_regions = [
        df_lsoa_regions.columns,                 # property
        ['any'] * len(df_lsoa_regions.columns),  # scenario
    ]
    # New DataFrame with the extra column level:
    df_lsoa_regions = pd.DataFrame(
        df_lsoa_regions.values,
        index=df_lsoa_regions.index,
        columns=cols_df_lsoa_regions
    )
    # Index column: 'region'.

    # Merge region info into LSOA data:
    df_lsoa = pd.merge(df_lsoa.copy(), df_lsoa_regions,
                       left_index=True, right_index=True, how='left')

    # Reset index for easier column selection:
    df_regions = df_regions.reset_index()

    # Input dataframes should contain multiple scenarios in a
    # MultiIndex column heading. Pick out each one in turn
    # and calculate the periphery units and regions.
    for scenario in scenario_list:
        # Names of selected LSOA:

        mask = (df_lsoa[('selected', scenario)] == 1)
        lsoa_selected = list(
            df_lsoa.loc[mask].copy().reset_index()['lsoa_code'])

        # Names of selected units:
        units_selected = df_units[
            df_units[('selected', scenario)] == 1].index.values
        # Regions containing selected units:
        regions_selected = df_units[
            df_units[('selected', scenario)] == 1]['region_code'].values.flatten()

        # Which columns do we want?
        cols = df_lsoa.columns.get_level_values('scenario').isin(
            ['any', scenario])
        # Subset of only these columns:
        df_lsoa_here = df_lsoa.loc[:, cols].copy()
        # # Drop the 'scenario' level
        df_lsoa_here = df_lsoa_here.droplevel('scenario', axis='columns')

        d = link_pathway_geography(
            df_lsoa_here, df_units,
            units_selected, lsoa_selected
            )

        # Add these results to the starting dataframes:
        df_units[('periphery_unit', scenario)] = 0
        mask = df_units.index.isin(d['periphery_units'])
        df_units.loc[mask, ('periphery_unit', scenario)] = 1

        df_regions[('contains_unit', scenario)] = 0
        mask = df_regions['region_code'].isin(regions_selected)
        df_regions.loc[mask, ('contains_unit', scenario)] = 1

        df_regions[('contains_periphery_lsoa', scenario)] = 0
        mask = df_regions['region_code'].isin(
            d['regions_containing_lsoa'])
        df_regions.loc[mask, ('contains_periphery_lsoa', scenario)] = 1

        df_regions[('contains_periphery_unit', scenario)] = 0
        mask = df_regions['region_code'].isin(
            d['regions_with_periphery_units'])
        df_regions.loc[mask, ('contains_periphery_unit', scenario)] = 1

    # Set index back to how it was earlier:
    df_regions = df_regions.set_index(['region', 'region_code'])
    return df_regions, df_units


def link_pathway_geography(
        df_lsoa, df_units,
        units_selected, lsoa_selected
        ):
    """
    Find:
    + regions containing selected LSOA
    + regions containing selected units
    + LSOA in regions containing selected units
    + stroke units catching those LSOA (periphery units)
    + regions containing periphery units

    periphery units catching any LSOA in selected regions.

    TO DO - what about sometimes combined, sometimes not files? ---------------------
    # Run the following on the "any" part of the dataframes only.

    """
    # Mask for selected LSOA:
    mask_lsoa_selected = df_lsoa.copy().reset_index()['lsoa_code'].isin(lsoa_selected).values
    # Find list of regions containing LSOA caught by selected units.
    regions_containing_lsoa = sorted(list(set(
        df_lsoa.loc[mask_lsoa_selected, 'region_code'])))

    # Reset index for easier access to values:
    df_units = df_units.copy().reset_index()
    # Remove 'scenario' column level:
    # (only need columns that belong in "any" scenario)
    df_units = df_units.droplevel('scenario', axis='columns')

    # Mask for selected units:
    mask_units_selected = df_units['postcode'].isin(units_selected)
    # Find list of regions containing selected units:
    regions_containing_units = list(df_units['region_code'][mask_units_selected])

    # Mask for LSOA in regions containing selected units:
    mask_lsoa_in_regions_containing_units = (
        df_lsoa['region_code'].isin(regions_containing_units))
    # Find list of periphery units:
    periphery_units = sorted(list(set(
        df_lsoa.loc[mask_lsoa_in_regions_containing_units, 'unit_postcode'])))

    # Mask for regions containing periphery units:
    mask_regions_periphery_units = (
        df_units['postcode'].isin(periphery_units))
    # Find list of periphery regions:
    regions_with_periphery_units = list(
        df_units.loc[mask_regions_periphery_units, 'region_code'])

    to_return = {
        'periphery_units': periphery_units,
        'regions_containing_lsoa': regions_containing_lsoa,
        'regions_with_periphery_units': regions_with_periphery_units
    }
    return to_return


# ########################
# ##### COMBINE DATA #####
# ########################
def make_all_geometry_data(
        df_lsoa, df_units, df_regions,
        df_transfer=None, df_lsoa_results=None
        ):

    df_regions = load_regions()
    df_regions, df_units = make_new_periphery_data(
        df_regions, df_units, df_lsoa)

    gdf_boundaries_regions = _load_geometry_regions(df_regions)
    gdf_points_units = _load_geometry_stroke_units(df_units)
    if df_transfer is None:
        gdf_lines_transfer = pd.DataFrame()  # Leave blank.
    else:
        gdf_lines_transfer = _load_geometry_transfer_units(df_transfer)
    gdf_boundaries_lsoa = _load_geometry_lsoa(df_lsoa, df_lsoa_results)

    # Merge many LSOA into one big blob of catchment area.
    gdf_boundaries_catchment = _load_geometry_catchment(gdf_boundaries_lsoa)

    # Only keep separate LSOA that have been selected.
    df_select = gdf_boundaries_lsoa.xs(
        'selected', axis='columns', level='property', drop_level=False)
    mask = (df_select == 1).any(axis='columns')
    gdf_boundaries_lsoa = gdf_boundaries_lsoa.loc[mask].copy()

    # # Make columns in regions and transfer that can be used for any
    # # scenario created in the LSOA data using diff.
    # # e.g. 'diff_drip-and-ship_minus_mothership' scenario wasn't
    # # run directly in the pathway and so its regions and units info
    # # doesn't exist yet.
    # scenario_list = sorted(list(set(
    #     gdf_boundaries_lsoa.columns.get_level_values('scenario'))))
    # scenario_list.remove('any')
    # for scenario in scenario_list:
    #     if 'diff' in scenario:
    #         # Add any other columns that these expect.
    #         gdf_boundaries_regions = create_combo_cols(
    #             gdf_boundaries_regions, scenario)
    #         gdf_points_units = create_combo_cols(
    #             gdf_points_units, scenario)
    #     else:
    #         # The data for non-diff scenarios should already exist.
    #         pass
    # TO DO - fixing to create_combo_cols required ----------------------------------

    # For each gdf, reset the index so that the index columns
    # appear in a saved .geojson and label the new index column.
    def make_new_index(gdf):
        gdf = gdf.reset_index()
        gdf.index.name = 'id'
        return gdf

    gdf_boundaries_regions = make_new_index(gdf_boundaries_regions)
    gdf_points_units = make_new_index(gdf_points_units)
    gdf_lines_transfer = make_new_index(gdf_lines_transfer)
    gdf_boundaries_lsoa = make_new_index(gdf_boundaries_lsoa)
    gdf_boundaries_catchment = make_new_index(gdf_boundaries_catchment)

    to_return = (
        gdf_boundaries_regions,
        gdf_points_units,
        gdf_lines_transfer,
        gdf_boundaries_lsoa,
        gdf_boundaries_catchment
    )

    return to_return


def _load_geometry_regions(df_regions):
    """
    Create GeoDataFrames of new geometry and existing DataFrames.
    """
    # All region polygons:
    gdf_list = []
    gdf_boundaries_regions_e = import_geojson('SICBL22NM')
    gdf_list.append(gdf_boundaries_regions_e)
    gdf_boundaries_regions_w = import_geojson('LHB20NM')
    gdf_list.append(gdf_boundaries_regions_w)
    # Combine:
    gdf_boundaries_regions = pd.concat(gdf_list, axis='rows')

    # Index column: 'region'.
    # Always has only one unnamed column index level.

    # Drop columns that appear in both DataFrames:
    gdf_boundaries_regions = gdf_boundaries_regions.drop(
        'region', axis='columns'
    )

    # ----- Prepare separate data -----
    # Set up column level info for the merged DataFrame.
    # Everything needs two levels: scenario, property.
    # Geometry:
    cols_gdf_boundaries_regions = [
        gdf_boundaries_regions.columns,                 # property
        ['any'] * len(gdf_boundaries_regions.columns),  # scenario
    ]
    # Final data:
    col_level_names = ['property', 'scenario']
    col_geometry = ('geometry', 'any')

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

    # Sort the results by scenario:
    gdf_boundaries_regions = gdf_boundaries_regions.sort_index(
        axis='columns', level='scenario')

    # Convert to GeoDataFrame:
    gdf_boundaries_regions = geopandas.GeoDataFrame(
        gdf_boundaries_regions,
        geometry=col_geometry
        )

    return gdf_boundaries_regions


def _load_geometry_stroke_units(df_units):
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
    # Index column: Postcode.
    # Expected column MultiIndex levels:
    #   - combined: ['scenario', 'property']
    #   - separate: ['{unnamed level}']

    # Load and parse geometry data
    # TO DO - change to relative import.
    # # Relative import from package files:
    # path_to_file = files('map.data').joinpath('unit_postcodes_coords.csv')
    path_to_file = os.path.join('data', 'unit_postcodes_coords.csv')
    df_coords = pd.read_csv(path_to_file, index_col='postcode')
    # Index: postcode.
    # Columns: BNG_E, BNG_N, Longitude, Latitude.
    # Add another column level to the coordinates.
    cols_df_coords = [
        df_coords.columns,                 # property
        ['any'] * len(df_coords.columns),  # scenario
    ]
    df_coords = pd.DataFrame(
        df_coords.values,
        index=df_coords.index,
        columns=cols_df_coords
    )
    # Merge:
    df_units = pd.merge(
        df_units, df_coords, left_index=True, right_index=True, how='left')

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

    return gdf_units


def _load_geometry_transfer_units(df_transfer):
    """
    Create GeoDataFrames of new geometry and existing DataFrames.
    """
    # Selected stroke units names, coordinates, and services.
    # Index column: ['postcode', 'name_nearest_mt']
    # Expected column MultiIndex levels:
    #   - combined: ['scenario', 'property']
    #   - separate: ['{unnamed level}']

    # Load and parse geometry data
    # TO DO - change to relative import.
    # # Relative import from package files:
    # path_to_file = files('map.data').joinpath('unit_postcodes_coords.csv')
    path_to_file = os.path.join('data', 'unit_postcodes_coords.csv')
    df_coords = pd.read_csv(path_to_file)
    # Columns: postcode, BNG_E, BNG_N, Longitude, Latitude.

    # From the loaded file:
    x_col = 'BNG_E'
    y_col = 'BNG_N'
    x_col_mt = 'BNG_E_mt'
    y_col_mt = 'BNG_N_mt'
    # To be created here:
    col_unit = 'unit_coords'
    col_tran = 'transfer_coords'
    col_line_coords = ('line_coords', 'any')
    col_line_geometry = ('geometry', 'any')

    # DataFrame of just the arrival and transfer units:
    df_arrival_transfer = df_transfer.index.to_frame(index=False)
    # If there are multiple column levels, only keep the lowest.
    if 'scenario' in df_arrival_transfer.columns.names:
        # TO DO - drop by level name. --------------------------------------------------
        df_arrival_transfer = (
            df_arrival_transfer.droplevel('scenario', axis='columns'))
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
        left_on='transfer_unit_postcode', right_on='postcode',
        how='left', suffixes=(None, '_mt')
        )
    df_arrival_transfer = m2.drop(['postcode_mt'], axis='columns')
    # Set the index columns to match the main DataFrame's:
    df_arrival_transfer = df_arrival_transfer.set_index(
        ['postcode', 'transfer_unit_postcode'])
    # Index: 'postcode', 'name_nearest_mt'
    # Columns: BNG_E, BNG_N, Longitude, Latitude,
    #          BNG_E_mt, BNG_N_mt, Longitude_mt, Latitude_mt

    # Add another column level to the coordinates.
    cols_df_arrival_transfer = [
        df_arrival_transfer.columns,                 # property
        ['any'] * len(df_arrival_transfer.columns),  # scenario
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
    gdf_transfer = create_lines_from_coords(
        df_transfer,
        [col_unit, col_tran],
        col_line_coords,
        col_line_geometry
        )

    return gdf_transfer


def _load_geometry_lsoa(df_lsoa, df_results_by_lsoa=None):
    """
    Create GeoDataFrames of new geometry and existing DataFrames.

        df_lsoa_results = df_results_by_lsoa
        # Index column: lsoa.
        # Expected column MultiIndex levels:
        #   - combined: ['scenario', 'property', 'subtype']
        #   - separate: ['property', 'subtype]
    """

    # All LSOA shapes:
    gdf_boundaries_lsoa = import_geojson('LSOA11NM')
    # Index column: LSOA11CD.
    # Always has only one unnamed column index level.
    gdf_boundaries_lsoa = gdf_boundaries_lsoa.reset_index()
    gdf_boundaries_lsoa = gdf_boundaries_lsoa.rename(columns={'LSOA11NM': 'lsoa', 'LSOA11CD': 'lsoa_code'})
    gdf_boundaries_lsoa = gdf_boundaries_lsoa.set_index(['lsoa', 'lsoa_code'])

    # TO DO - shorten LSOA geojson load time by separating big file into multiple smaller ones,
    # e.g. one file per region of the UK (South West, West Midlands...)
    # or one file per SICBL / LHB (too many files?)
    # then work out which ones should be loaded and only load those ones.
    # Will almost never want to load in Newcastle LSOA when we're looking at Cornwall.

    # Results by LSOA.
    if df_results_by_lsoa is None:
        results_exist = False
    else:
        results_exist = True

    # ----- Prepare separate data -----
    # Set up column level info for the merged DataFrame.
    # Everything needs three levels: scenario, property, subtype.
    # LSOA names:
    cols_lsoa = df_lsoa.columns
    df_lsoa_column_arr = np.array(
        [[n for n in c] for c in cols_lsoa])
    cols_df_lsoa = [
        df_lsoa_column_arr[:, 0],                    # property
        df_lsoa_column_arr[:, 1],                    # scenario
        [''] * len(cols_lsoa),                       # subtype
    ]
    # Geometry:
    cols_gdf_boundaries_lsoa = [
        gdf_boundaries_lsoa.columns,                 # property
        ['any'] * len(gdf_boundaries_lsoa.columns),  # scenario
        [''] * len(gdf_boundaries_lsoa.columns),     # subtype
    ]
    # Final data:
    col_level_names = ['property', 'scenario', 'subtype']
    col_geometry = ('geometry', 'any', '')

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
            gdf_boundaries_lsoa, df_results_by_lsoa,
            left_index=True, right_index=True, how='left'
        )
    # Name the column levels:
    gdf_boundaries_lsoa.columns = (
        gdf_boundaries_lsoa.columns.set_names(col_level_names))

    # Sort the results by scenario:
    gdf_boundaries_lsoa = gdf_boundaries_lsoa.sort_index(
        axis='columns', level='scenario')

    # Convert to GeoDataFrame:
    gdf_boundaries_lsoa = geopandas.GeoDataFrame(
        gdf_boundaries_lsoa,
        geometry=col_geometry
        )

    return gdf_boundaries_lsoa


def _load_geometry_catchment(gdf_boundaries_lsoa):
    # List of scenarios included in the LSOA data:
    scenario_list = sorted(list(set(
        gdf_boundaries_lsoa.columns.get_level_values('scenario'))))
    scenario_list.remove('any')

    # Store resulting polygons in here:
    dfs_to_merge = {}

    # For each scenario:
    for scenario in scenario_list:
        if scenario.startswith('diff'):
            pass
        else:
            col_to_dissolve = ('unit_postcode', scenario, '')
            col_geometry = ('geometry', 'any', '')
            df = _combine_lsoa_into_catchment_shapes(
                gdf_boundaries_lsoa,
                col_to_dissolve=col_to_dissolve,
                col_geometry=col_geometry,
                col_after_dissolve='unit'
                )
            # Which ones are selected?
            selected_units = gdf_boundaries_lsoa[col_to_dissolve][
                gdf_boundaries_lsoa[('selected', scenario)] == 1]
            df['selected'] = 0
            mask = df.index.isin(selected_units)
            df.loc[mask, 'selected'] = 1
            # Which ones are used in this scenario?
            df['use'] = 1

            # Set index column:
            df = df.reset_index()
            df = df.set_index(['unit', 'geometry'])
            # Store in the main list:
            dfs_to_merge[scenario] = df

    # Can't concat without index columns.
    gdf_boundaries_catchment = pd.concat(
        dfs_to_merge.values(),
        axis='columns',
        keys=dfs_to_merge.keys()  # Names for extra index row
        )
    # The combo dataframe contains only columns for scenario / property,
    # so switch them round to property / scenario:
    gdf_boundaries_catchment.columns = gdf_boundaries_catchment.columns.swaplevel(0, 1)
    # Rename index so it can be made into a normal column:
    gdf_boundaries_catchment = gdf_boundaries_catchment.rename(
        index={gdf_boundaries_catchment.index.name:(col_to_dissolve)})
    gdf_boundaries_catchment = gdf_boundaries_catchment.reset_index()
    # Give the new, useless index a name.
    gdf_boundaries_catchment.index.name = 'id'

    col_level_names = ['property', 'scenario']
    # Name the column levels:
    gdf_boundaries_catchment.columns = (
        gdf_boundaries_catchment.columns.set_names(col_level_names))

    # Set geometry column:
    gdf_boundaries_catchment = gdf_boundaries_catchment.set_geometry(
        'geometry')

    return gdf_boundaries_catchment


def _combine_lsoa_into_catchment_shapes(
        gdf, col_to_dissolve, col_geometry='geometry',
        col_after_dissolve='dissolve'):
    """
    # Combine LSOA geometry - from separate polygon per LSOA to one
    # big polygon for all LSOAs in catchment area.
    """
    # Copy to avoid pandas shenanigans.
    gdf = gdf.copy()
    # Assuming that index contains stuff to be dissolved,
    # it doesn't make sense to keep it afterwards.
    # Make it a normal column so it can be neglected.
    gdf = gdf.reset_index()
    # Make a fresh dataframe with no column multiindex:
    gdf2 = pd.DataFrame(
        gdf[[col_to_dissolve, col_geometry]].values,
        columns=[col_after_dissolve, 'geometry']
    )
    gdf2 = geopandas.GeoDataFrame(gdf2, geometry='geometry')
    gdf2 = gdf2.dissolve(by=col_after_dissolve)
    return gdf2


# ############################
# ##### HELPER FUNCTIONS #####
# ############################
def create_lines_from_coords(
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


def create_combo_cols(gdf, scenario):
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

    # TO DO - this needs big fixing now that 'scenario' level isn't at the top -----------------------

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


# ###################
# ##### COLOURS #####
# ###################
def assign_colours_to_regions(gdf, region_type, col_col):
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


def assign_colours_to_regions_BROKEN(gdf, region_type):

    colours = ['ForestGreen', 'LimeGreen', 'RebeccaPurple', 'Teal']

    # TO DO - this neighbours function isn't working right -----------------------------
    # currently says that Gloucestershire doesn't border Bristol or Bath regions -------
    gdf = find_neighbours_for_regions(gdf, region_type)
    gdf = gdf.sort_values('total_neighbours', ascending=False)

    neighbour_list = gdf[region_type].tolist()

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


def round_coordinates(df_geojson):
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


def find_neighbours_for_regions(df_geojson, region_type='ICG22NM'):
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

    df_geojson = round_coordinates(df_geojson)
    df_new = split_multipolygons(df_geojson)
    df_neighbours = find_neighbours(df_new, region_type)
    df = unsplit_multipolygons(df_geojson, df_neighbours)

    # Record the number of neighbours:
    df['total_neighbours'] = df['neighbour_list'].str.len()
    return df
