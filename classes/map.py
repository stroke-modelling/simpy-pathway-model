"""
Stick this in a new home eventually. Not a class yet.
"""

import numpy as np
import pandas as pd
import geopandas
import matplotlib.pyplot as plt
import os

def plot_map_selected_units(setup, col='ICB22NM'):
    # Select geojson file based on input "col":
    geojson_file_dict = {
        'CCG19NM': setup.file_geojson_ccg,
        'ICB22NM': setup.file_geojson_icb,
        'LAD17NM': setup.file_geojson_lad,
        'STP19NM': setup.file_geojson_stp,
        'LHB20NM': setup.file_geojson_lhb,
        'SCN17NM': setup.file_geojson_scn,
        'RGN11NM': setup.file_geojson_rgn,
    }
    # Import region file:
    dir_input = setup.dir_data_geojson
    file_input = geojson_file_dict[col]
    path_to_file = os.path.join(dir_input, file_input)
    gdf_boundaries = geopandas.read_file(path_to_file)#, crs='EPSG:27700')

    # Import selected stroke unit data:
    dir_input = setup.dir_output
    file_input = setup.file_selected_stroke_units
    path_to_file = os.path.join(dir_input, file_input)
    df_units = pd.read_csv(path_to_file)

    # Create coordinates:
    # Current setup means sometimes these columns have different names.
    # TO DO - fix that please! ---------------------------------------------------
    try:
        long = df_units.Easting
        lat = df_units.Northing
    except AttributeError:
        long = df_units.long_x
        lat = df_units.lat_x
    df_units['geometry'] = geopandas.points_from_xy(long, lat)

    # Convert to GeoDataFrame:
    gdf_units = geopandas.GeoDataFrame(
        df_units, geometry=df_units['geometry']#, crs="EPSG:4326"
    )

    # # Pick out the region names with repeats:
    # regions = df_units[col].copy()
    # # Remove missing values:
    # regions = regions.dropna()
    # # Remove repeats:
    # regions = list(set(regions))

    # Restrict the geojson data to only these regions:
    # mask = [gdf[col].str.contains(s) for s in regions]
    # mask = np.any(mask, axis=0)
    # gdf = gdf[mask]
    gdf_boundaries = pd.merge(
        gdf_boundaries, gdf_units[col],
        left_on=col, right_on=col,
        how='right'
    )

    # Plot the map.
    fig, ax = plt.subplots(figsize=(10, 10))  # Make max dimensions XxY inch

    # Draw the main map with colours (choropleth):
    gdf_boundaries.plot(
        ax=ax,              # Set which axes to use for plot (only one here)
        antialiased=False,  # Avoids artifact boundry lines
        # facecolor='none',
        column=gdf_boundaries.index.name,
        cmap='viridis',
        edgecolor='k',   # Make LSOA boundry same colour as area
        )

    # Scatter marker for each hospital:
    gdf_units.plot(
        ax=ax,
        edgecolor='k',
        facecolor='w',
        markersize=50,
        marker='o'
        )

    # Scatter marker star for MT/MSU units:
    mask = gdf_units['Use_MT'] == 1
    MSU = gdf_units[mask]
    MSU.plot(
        ax=ax,
        edgecolor='k',
        facecolor='y',
        markersize=300,
        marker='*'
        )

    # Draw a line connecting each unit to its MT unit.
    # ... TO DO! -------------------------------------------------------------------

    # Add labels
    # Define "z" to shorten following "for" line:
    z = zip(
        gdf_units.geometry.x,
        gdf_units.geometry.y,
        gdf_units.Hospital_name
        )
    for x, y, label in z:
        # Edit the label to put a space in the postcode when displayed:
        label = f'{label[:-3]} {label[-3:]}'
        # Place the label slightly offset from the
        # exact hospital coordinates (x, y).
        ax.annotate(
            label, xy=(x, y), xytext=(8, 8),
            textcoords="offset points",
            backgroundcolor="w", fontsize=8
            )

    ax.set_axis_off()  # Turn off axis line and numbers

    # Save output to output folder.
    dir_output = setup.dir_output
    file_name = setup.file_selected_units_map
    path_to_file = os.path.join(dir_output, file_name)
    plt.savefig(path_to_file, dpi=300, bbox_inches='tight')
    plt.close()
