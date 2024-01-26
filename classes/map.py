"""
Stick this in a new home eventually. Not a class yet.

crs reference:
+ EPSG:4326  - longitude / latitude.
+ CRS:84     - same as EPSG:4326.
+ EPSG:27700 - British National Grid (BNG).
"""
import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import os

from shapely import LineString  # For creating line geometry.


# ##########################
# ##### DATA WRANGLING #####
# ##########################
def import_geojson(setup: 'Setup', region_type: 'str'):
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
        'LSOA11NM': setup.file_geojson_lsoa,
        'CCG19NM': setup.file_geojson_ccg,
        'ICB22NM': setup.file_geojson_icb,
        'LAD17NM': setup.file_geojson_lad,
        'STP19NM': setup.file_geojson_stp,
        'LHB20NM': setup.file_geojson_lhb,
        'SCN17NM': setup.file_geojson_scn,
        'RGN11NM': setup.file_geojson_rgn,
    }
    # n.b. current setup as of January 2024 is that the dict
    # keys match the column names in the LSOA_regions.csv
    # and similar reference files. The actual geojson files
    # definitely contain the same type of region, but could
    # be from a different year than the one listed here.

    # Import region file:
    dir_input = setup.dir_data_geojson
    file_input = geojson_file_dict[region_type]
    path_to_file = os.path.join(dir_input, file_input)
    gdf_boundaries = geopandas.read_file(path_to_file)
    # If crs is given in the file, geopandas automatically
    # pulls it through. Convert to National Grid coordinates:
    if gdf_boundaries.crs != 'EPSG:27700':
        gdf_boundaries = gdf_boundaries.to_crs('EPSG:27700')
    return gdf_boundaries


def import_selected_stroke_units(setup: 'Setup'):
    """
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
    # Import selected stroke unit data:
    dir_input = setup.dir_output
    file_input = setup.file_selected_stroke_units
    path_to_file = os.path.join(dir_input, file_input)
    df_units = pd.read_csv(path_to_file)

    # Create coordinates:
    # Current setup means sometimes these columns have different names.
    # TO DO - fix that please! ---------------------------------------------------
    x = df_units['Easting']
    y = df_units['Northing']
    xy = df_units[['Easting', 'Northing']]
    crs = 'EPSG:27700'

    df_units['geometry'] = geopandas.points_from_xy(x, y)
    # Make a column of coordinates [x, y]:
    df_units['coords'] = xy.values.tolist()

    # Convert to GeoDataFrame:
    gdf_units = geopandas.GeoDataFrame(
        df_units, geometry=df_units['geometry'], crs=crs
    )
    # Convert to British National Grid coordinates if necessary:
    if crs != 'EPSG:27700':
        gdf_units = gdf_units.to_crs('EPSG:27700')
    return gdf_units


def import_transfer_unit_data(setup: 'Setup'):
    """
    Import national transfer unit data.

    The file read is the output from Units() after the
    national stroke unit services have been updated.

    Inputs
    ------
    setup - Setup() object. Contains attributes for paths to the
            data directory and the geojson file names.

    Returns
    -------
    df - pd.DataFrame. One row per national stroke unit in the file.
         Columns are each stroke unit, its time to chosen MT unit,
         and the name of its chosen MT unit:
         ['from_postcode', 'time_nearest_MT', 'name_nearest_MT'].
    """
    dir_input = setup.dir_output
    file_input = setup.file_national_transfer_units
    path_to_file = os.path.join(dir_input, file_input)
    df = pd.read_csv(path_to_file)
    return df


def import_selected_lsoa(setup: 'Setup'):
    """
    Import data on selected LSOAs.

    Inputs
    ------
    setup - Setup() object. Contains attributes for paths to the
            data directory and the geojson file names.

    Returns
    -------
    df - pd.DataFrame. Names, codes, coordinates of selected LSOAs.
         ['LSOA11NM', 'LSOA11CD', 'LSOA11BNG_N', 'LSOA11BNG_E',
          'LSOA11LONG', 'LSOA11LAT']
    """
    dir_input = setup.dir_output
    file_input = setup.file_selected_lsoas
    path_to_file = os.path.join(dir_input, file_input)
    df = pd.read_csv(path_to_file)
    return df


def import_lsoa_travel_data(setup: 'Setup'):
    """
    Import each LSOA's chosen stroke units and travel times.

    Inputs
    ------
    setup - Setup() object. Contains attributes for paths to the
            data directory and the geojson file names.

    Returns
    -------
    df - pd.DataFrame. Each LSOA's name, chosen IVT/MT/MSU units,
         and travel times to those units. For X in IVT, MT, MSU:
         ['LSOA11NM', 'LSOA11CD', 'time_nearest_X',
          'postcode_nearest_X', 'ssnap_name_nearest_X']
    """
    dir_input = setup.dir_output
    file_input = setup.file_national_lsoa_travel
    path_to_file = os.path.join(dir_input, file_input)
    df = pd.read_csv(path_to_file)
    return df


def keep_only_selected_units(
        df: 'pd.DataFrame', df_units: 'pd.DataFrame',
        left_col: 'str', right_col: 'str', how: 'str' = 'right'
        ):
    """
    Wrapper for pd.merge() for limiting one df to rows in another.

    This function exists just so that the effect of it is clear
    from the function name without needing comments.

    Inputs
    ------
    df        - pd.DataFrame. Contains rows for many things we're
                not interested in.
    df_units  - pd.DataFrame. Contains only rows of things we're
                interested in.
    left_col  - str. Name of the column of "df" to match.
    right_col - str. Name of the column of "df_units" to match,
                and the only column whose data will be kept in
                the resulting DataFrame.
    how       - str. How to merge the DataFrames. Normally want
                'right' to keep only the rows of df_units and
                remove any row in df that is not in df_units.

    Returns
    -------
    df - pd.DataFrame. Reduced DataFrame.
    """
    df = pd.merge(
        df, df_units[right_col],
        left_on=left_col, right_on=right_col,
        how=how
    )
    return df


def copy_columns_from_dataframe(
        df_left: 'pd.DataFrame',
        df_right: 'pd.DataFrame',
        cols_to_copy: list,
        left_col: 'str', right_col: 'str', how: 'str' = 'right',
        cols_to_rename_dict: dict = {},
        ):
    """
    Copy data from one DataFrame into another. Wrapper for pd.merge().

    This function exists just so that the effect of it is clear
    from the function name without needing comments.

    Inputs
    ------
    df_left             - pd.DataFrame. Contains rows for many things
                          we're not interested in.
    df_right            - pd.DataFrame. Contains only rows of things
                          we're interested in.
    cols_to_copy        - list. Names of the columns from df_right
                          to copy over.
    left_col            - str. Name of the column of df_left to match.
    right_col           - str. Name of the column of df_right to match.
    how                 - str. How to merge the DataFrames.
    cols_to_rename_dict - dict. Rename the columns in df_left that
                          have been copied over from df_right.

    Returns
    -------
    df_left - pd.DataFrame. Same as the input df_left with the addition
              of copied columns from df_right.
    """
    # Copy data from df_right to df_left:
    df_left = pd.merge(
        df_left,
        df_right[[right_col] + cols_to_copy],
        left_on=left_col, right_on=right_col,
        how=how
    )

    # Rename columns in df_left:
    df_left = df_left.rename(columns=cols_to_rename_dict)
    return df_left


def create_lines_from_coords(df, cols_with_coords):
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
    df['line_coords'] = df[cols_with_coords].values.tolist()

    # Convert line coords to LineString objects:
    df['line_geometry'] = [
        LineString(coords) for coords in df['line_coords']]

    # Convert to GeoDataFrame:
    gdf = geopandas.GeoDataFrame(
        df, geometry=df['line_geometry']#, crs="EPSG:4326"
    )
    # TO DO - implement CRS explicitly ---------------------------------------------
    return gdf


# ####################
# ##### PLOTTING #####
# ####################
# n.b. The following functions mostly just use plt.plot()
# but are given different wrappers anyway for the sake of
# applying some kwargs automatically.
def draw_boundaries(ax, gdf, **kwargs):
    """
    Draw regions from a GeoDataFrame.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the regions.
    gdf    - GeoDataFrame. Stores geometry to be plotted.
    kwargs - dict. Keyword arguments to pass to plt.plot().

    Returns
    -------
    ax - pyplot axis. Same as input but with regions drawn on.
    """
    # Draw the main map with colours (choropleth):
    gdf.plot(
        ax=ax,              # Set which axes to use for plot (only one here)
        antialiased=False,  # Avoids artifact boundry lines
        **kwargs
        )
    return ax


def scatter_units(ax, gdf):
    """
    Draw scatter markers for stroke units.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores stroke unit coordinates and services.

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    # Scatter marker for each hospital:
    gdf.plot(
        ax=ax,
        edgecolor='k',
        facecolor='w',
        markersize=50,
        marker='o',
        zorder=2
        )

    # TO DO - split off MT and MSU better -----------------------------------------------
    # Scatter marker star for MT/MSU units:
    mask = gdf['Use_MT'] == 1
    MSU = gdf[mask]
    MSU.plot(
        ax=ax,
        edgecolor='k',
        facecolor='y',
        markersize=300,
        marker='*',
        zorder=2
        )
    return ax


def plot_lines_between_units(ax, gdf):
    """
    Draw lines from stroke units to their MT transfer units.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores LineString objects that connect
             each stroke unit to its MT transfer unit.

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    # Draw a line connecting each unit to its MT unit.
    gdf.plot(
        ax=ax,
        edgecolor='k',
        linestyle='-',
        linewidth=3,
        zorder=1  # Place it beneath the scatter markers.
    )
    return ax


def annotate_unit_labels(ax, gdf):
    """
    Draw label for each stroke unit.

    Inputs
    ------
    ax     - pyplot axis. Where to draw the scatter markers.
    gdf    - GeoDataFrame. Stores coordinates and name of each
             stroke unit.

    Returns
    -------
    ax - pyplot axis. Same as input but with scatter markers.
    """
    # Define "z" to shorten following "for" line:
    z = zip(
        gdf.geometry.x,
        gdf.geometry.y,
        gdf.Hospital_name
        )
    for x, y, label in z:
        # Edit the label to put a space in the postcode when displayed:
        label = f'{label[:-3]} {label[-3:]}'
        # Place the label slightly offset from the
        # exact hospital coordinates (x, y).
        ax.annotate(
            label, xy=(x, y), xytext=(8, 8),
            textcoords="offset points",
            bbox=dict(facecolor='w', edgecolor='k'),
            fontsize=8
            )
    return ax


# ######################
# ##### MAIN PLOTS #####
# ######################
def plot_map_selected_units(setup, col='ICB22NM'):
    """
    Make map of the selected units and the regions containing them.

    Properties of this map:
    + Each stroke unit is shown with a scatter marker.
    + Non-MT units are shown as circles and MT units as stars.
    + Lines are drawn between each non-MT unit and its chosen MT unit.
    + Each stroke unit is labelled in an offset text box.
    + The regions that contain the selected units are drawn in
      the background with each region given a different colour from
      its neighbours. These regions have an outline.

    Required data files:
    + geojson file of choice.
      Must contain:
      + coordinates of each feature / region boundary shape.
    + selected stroke unit file
      Output from Scenario.
      Must contain:
      + Postcode
        - for unit name matching.
        - for labels on the map.
      + Use_MT
        - for scatter marker choice.
      + [region]
        - region names to match the geojson file, for limiting the
          plotted areas to just those containing the stroke units.
      + Easting, Northing
        - for placement of the scatter markers.
    + national transfer unit file
      Output from Units.
      + from_postcode
        - for unit name matching.
      + name_nearest_MT
        - for setting up lines drawn between each stroke unit and
          its nearest MT unit.

    Result is saved as the name given in setup.file_selected_units_map.
    """
    # ----- Setup -----
    # Import stroke unit data:
    gdf_units = import_selected_stroke_units(setup)
    df_transfer = import_transfer_unit_data(setup)
    df_transfer = keep_only_selected_units(
        df_transfer, gdf_units,
        left_col='from_postcode', right_col='Postcode')

    # Gather stroke unit coordinates to make connecting lines:
    # Copy over coordinates of each stroke unit...
    df_transfer = copy_columns_from_dataframe(
        df_transfer, gdf_units, 
        cols_to_copy=['coords'],
        cols_to_rename_dict={'coords': 'unit_coords'},
        left_col='from_postcode', right_col='Postcode', how='right'
        )
    # ... and its MT transfer unit:
    df_transfer = copy_columns_from_dataframe(
        df_transfer, gdf_units,
        cols_to_copy=['coords'],
        cols_to_rename_dict={'coords': 'transfer_coords'},
        left_col='name_nearest_MT', right_col='Postcode', how='left')
    gdf_transfer = create_lines_from_coords(
        df_transfer, ['unit_coords', 'transfer_coords'])

    # Import background region shapes:
    gdf_boundaries = import_geojson(setup, col)
    gdf_boundaries = keep_only_selected_units(
        gdf_boundaries, gdf_units, left_col=col, right_col=col)

    # ----- Plotting -----
    # Plot the map.
    # Make max dimensions XxY inch:
    fig, ax = plt.subplots(figsize=(10, 10))

    ax = draw_boundaries(
        ax, gdf_boundaries,
        column=gdf_boundaries.index.name,  # For colour choice
        cmap='Blues', edgecolor='silver', linewidth=0.5)
    ax = scatter_units(ax, gdf_units)
    ax = plot_lines_between_units(ax, gdf_transfer)
    ax = annotate_unit_labels(ax, gdf_units)

    ax.set_axis_off()  # Turn off axis line and numbers

    # Save output to output folder.
    dir_output = setup.dir_output
    file_name = setup.file_selected_units_map
    path_to_file = os.path.join(dir_output, file_name)
    plt.savefig(path_to_file, dpi=300, bbox_inches='tight')
    plt.close()


def plot_map_catchment(setup, col='ICB22NM'):
    """
    Map the selected units, containing regions, and catchment areas.

    Creates three maps.
    + "Drip & Ship" - catchment area of each IVT unit.
    + "Mothership" - catchment area of each MT unit, no IVT units.
    + "MSU" - catchment area of each MSU unit.

    Properties of all maps:
    + Each stroke unit is shown with a scatter marker.
    + Non-MT units are shown as circles and MT units as stars.
    + Lines are drawn between each non-MT unit and its chosen MT unit.
    + Each stroke unit is labelled in an offset text box.
    + The regions that contain the selected units are drawn in
      the background with each region given a different colour from
      its neighbours. These regions have an outline.

    Required data files:
    + geojson file of choice.
      Must contain:
      + coordinates of each feature / region boundary shape.
    + selected stroke unit file
      Output from Scenario.
      Must contain:
      + Postcode
        - for unit name matching.
        - for labels on the map.
      + Use_MT
        - for scatter marker choice.
      + [region]
        - region names to match the geojson file, for limiting the
          plotted areas to just those containing the stroke units.
      + Easting, Northing
        - for placement of the scatter markers.
    + national transfer unit file
      Output from Units.
      + from_postcode
        - for unit name matching.
      + name_nearest_MT
        - for setting up lines drawn between each stroke unit and
          its nearest MT unit.
    + geojson file of LSOA boundaries.
      Must contain:
      + coordinates of each LSOA boundary.
    + selected LSOA name file.
      Must contain:
      + column LSOA11CD, one row per selected LSOA.
    + national LSOA travel data.
      Must contain:
      + column LSOA11CD for name matching.
      + postcode_nearest_IVT
      + postcode_nearest_MT
      + postcode_nearest_MSU

    Result is saved as the name given in setup class:
    + file_drip_ship_map
    + file_mothership_map
    + file_msu_map
    """
    # ----- Stroke unit setup -----
    gdf_units = import_selected_stroke_units(setup)

    # Find MT transfer units for plotting lines between units:
    df_transfer = import_transfer_unit_data(setup)
    df_transfer = keep_only_selected_units(
        df_transfer, gdf_units,
        left_col='from_postcode', right_col='Postcode')
    # Copy over coordinates of each stroke unit...
    df_transfer = copy_columns_from_dataframe(
        df_transfer, gdf_units, 
        cols_to_copy=['coords'],
        cols_to_rename_dict={'coords': 'unit_coords'},
        left_col='from_postcode', right_col='Postcode', how='right'
        )
    # ... and its MT transfer unit:
    df_transfer = copy_columns_from_dataframe(
        df_transfer, gdf_units,
        cols_to_copy=['coords'],
        cols_to_rename_dict={'coords': 'transfer_coords'},
        left_col='name_nearest_MT', right_col='Postcode', how='left')
    gdf_transfer = create_lines_from_coords(
        df_transfer, ['unit_coords', 'transfer_coords'])

    # Find regional boundaries for reference on the map:
    gdf_boundaries = import_geojson(setup, col)
    gdf_boundaries = keep_only_selected_units(
        gdf_boundaries, gdf_units, left_col=col, right_col=col)

    # ----- LSOA setup -----
    df_lsoa = import_selected_lsoa(setup)

    # Find LSOA boundaries:
    gdf_boundaries_lsoa = import_geojson(setup, 'LSOA11NM')
    gdf_boundaries_lsoa = keep_only_selected_units(
        gdf_boundaries_lsoa,
        df_lsoa, left_col='LSOA11CD', right_col='LSOA11CD')

    # Match LSOA with its chosen stroke unit.
    df_lsoa_travel = import_lsoa_travel_data(setup)
    df_lsoa_travel = keep_only_selected_units(
        df_lsoa_travel, df_lsoa, left_col='LSOA11CD', right_col='LSOA11CD')
    cols_to_keep = [
        'LSOA11CD', 'postcode_nearest_IVT',
        'postcode_nearest_MT', 'postcode_nearest_MSU'
        ]
    gdf_boundaries_lsoa = pd.merge(
        gdf_boundaries_lsoa,
        df_lsoa_travel[cols_to_keep],
        left_on='LSOA11CD', right_on='LSOA11CD',
    )

    # ----- Plotting setup -----
    data_dicts = {
        'Drip & Ship': {
            'file': setup.file_drip_ship_map,
            'boundary_kwargs': {
                'column': 'postcode_nearest_IVT',
                'cmap': 'Blues',
                'edgecolor': 'face'
                }
            },
        'Mothership': {
            'file': setup.file_mothership_map,
            'boundary_kwargs': {
                'column': 'postcode_nearest_MT',
                'cmap': 'Blues',
                'edgecolor': 'face'
                }
            },
        'MSU': {
            'file': setup.file_msu_map,
            'boundary_kwargs': {
                'column': 'postcode_nearest_MSU',
                'cmap': 'Blues',
                'edgecolor': 'face'
                }
            },
    }

    # ----- Actual plotting -----
    for model_type, data_dict in zip(data_dicts.keys(), data_dicts.values()):
        # Plot the map.
        # Make max dimensions XxY inch:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(model_type)

        ax = draw_boundaries(
            ax, gdf_boundaries_lsoa,
            **data_dict['boundary_kwargs']
            )
        ax = draw_boundaries(
            ax, gdf_boundaries,
            facecolor='none', edgecolor='k', linewidth=0.5
            )
        ax = scatter_units(ax, gdf_units)
        ax = plot_lines_between_units(ax, gdf_transfer)
        ax = annotate_unit_labels(ax, gdf_units)

        ax.set_axis_off()  # Turn off axis line and numbers

        # Save output to output folder.
        dir_output = setup.dir_output
        file_name = data_dict['file']
        path_to_file = os.path.join(dir_output, file_name)
        plt.savefig(path_to_file, dpi=300, bbox_inches='tight')
        plt.close()
