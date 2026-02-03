import datacube
import geopandas as gpd
from datetime import datetime, timedelta
import pandas as pd
import xarray as xr
import numpy as np
import re

import json
from typing import Any

import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

from dea_tools.bandindices import calculate_indices
from dea_tools.datahandling import load_ard
from dea_tools.spatial import xr_vectorize

dc = datacube.Datacube(app="Burnt_severity")

attribute_mapping = {'ignition_d': 'ignition_date',
                    'capt_metho': 'capt_method',
                     'date_retri': 'date_retrieved',
                    'date_proce': 'date_processed'}

attribute_list = ['fire_id',                                               
                    'fire_name',
                    'fire_type',
                    'ignition_date',
                    'capt_date',
                    'capt_method',
                    'state',
                    'agency']

date_atts = ['ignition_date',
             'capt_date']

severity_list_dict = {0 : 'Unburnt',
                      1 : 'Grass_extent',
                      2 : 'Low',
                      3 : 'Medium',
                      4 : 'High',
                      5 : 'Extreme',
                      6 : 'No_data'}

resolution = (-10, 10)
measurements = ['nbart_blue', 'nbart_green', 'nbart_red',
                'nbart_nir_1','nbart_nir_2', 'nbart_swir_2','nbart_swir_3','oa_nbart_contiguity','oa_s2cloudless_mask']

output_crs = 'EPSG:3577'




def find_latest_valid_pixel(dataset):
    """
    For a multi-temporal xarray Dataset (masked for clouds/contiguity),
    return the latest clear value for each pixel across all bands.
    Output: Dataset with same bands, no time dimension.
    """
    # Ensure 'time' is a dimension
    if 'time' not in dataset.dims:
        raise ValueError("Dataset must have a 'time' dimension")

    # Convert time to numeric for comparison
    time_numeric = xr.DataArray(dataset['time'].astype('datetime64[ns]').astype(np.int64), dims='time')

    # Create mask for valid values (non-NaN)
    valid_mask = ~np.isnan(dataset['nbart_blue'])

    # Multiply mask by time to get numeric time where valid, else 0
    valid_times = valid_mask * time_numeric

    # Find latest valid time for each pixel
    latest_valid_time = valid_times.max(dim='time')

    # Create mask for latest time
    latest_mask = valid_times == latest_valid_time

    # Select latest valid values for each band
    latest_values = dataset.where(latest_mask).max(dim='time')

    return latest_values

def clean_name(name: str) -> str:
    """this takes a tring and ensures it dosn't contain any punctuation, numbers or special characters. then replaces spaces with underscore
    this is for the purpose of turning long, annoying fire names into someing that can be used to save files with, no manual fussing requiered"""
    # Remove numbers and special characters (keep only letters and spaces)
    cleaned = re.sub(r'[^A-Za-z\s]', '', name)
    # Replace spaces with underscores
    cleaned = re.sub(r'\s+', '_', cleaned)
    return cleaned


def _to_iso_z(dt: datetime) -> str:
    # ISO
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def is_date_iso(date_str):
    """ checks to see if date string has desiered ios UTC format by trying to convert it to datetime object
    returns TRUE if string is in correct format
    returns FLASE if it is no :(
    
    """
    try:
        datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%fZ")
        return True
    except (ValueError, TypeError):
        return False


def is_date_DEA_format(date_str):
    # test for YYYY-MM-DD format date:
    try:
        datetime.strptime(date_str[0:10], "%Y-%m-%d")
        return True
    except (ValueError, TypeError):
        return False


def process_date(date):
    """check the format for a date is as desiered 'yyyy-MM-ddTHH:mm:ss.sZ ', If it is not then format assuming it's 
    a valid date either as YYYY-MM-DD, YYYYMMDD or YYYYMMDDhhmmss and format as above (assume the time is noon if no time is given)

    otherwise return None. """

    try:
        date_as_str = str(date)
        print(f'date as string is {date_as_str}')
    except:
        return
    
    if date_as_str == 'nan':
        return
        
    if date_as_str == '0':
        return
    
    elif is_date_iso(date_as_str) == True:
        return date
        
    elif is_date_DEA_format(date_as_str) == True:
        return f'{date_as_str[0:10]}T12:00:00.0Z'
        
    elif bool(re.match(r'^\d+$', date_as_str)) == True:
        if len(date_as_str) == 14:
            return f'{date[0:4]}-{date[4:6]}-{date[6:8]}T{date[8:10]}:{date[10:12]}:{date[12:14]}.0Z'
        else:
            return f'{date[0:4]}-{date[4:6]}-{date[6:8]}T12:00:00.0Z'
    else:
        try:
            as_date = datetime.strptime(date_as_str, "%Y%m%d%H%M%S.%f")
            return as_date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            
        except ValueError:
            return

def test_polygon_attributes(polygon):

    if list(polygon)[3] == 'ignition_d':
        print('attributes are short')
        #rename truncated colums with long from of attribute names
        polygon = polygon.rename(columns= attribute_mapping)
        print('making attributes long')
        return polygon
    else:
        print('attibutes are long, no need to change')
        return polygon

def display_geometry_on_map(geom, zoom_bias=0):
    """
    Plots a datacube.utils.geometry.Geometry object on a folium map.

    Parameters
    ----------
    geom : datacube.utils.geometry.Geometry
        A geometry object to plot.
    zoom_bias : int or float
        Optional zoom adjustment.

    Returns
    -------
    folium.Map
        Interactive map with the geometry overlay.
    """
    import folium
    import numpy as np
    from pyproj import Transformer
    
    # Transform geometry to EPSG:4326 if needed
    if geom.crs.to_epsg() != 4326:
        geom = geom.to_crs('EPSG:4326')

    # Prepare list of polygons
    polygons = []
    if geom.geom.geom_type == 'Polygon':
        polygons.append(list(geom.geom.exterior.coords))
    elif geom.geom.geom_type == 'MultiPolygon':
        for poly in geom.geom.geoms:
            polygons.append(list(poly.exterior.coords))
    else:
        raise ValueError(f"Unsupported geometry type: {geom.geom.geom_type}")

    # Flatten all coordinates for center calculation
    all_coords = [coord for poly in polygons for coord in poly]
    lat_lon_coords = [(y, x) for x, y in all_coords]  # Folium expects (lat, lon)

    # Calculate center and zoom
    lats, lons = zip(*lat_lon_coords)
    center = [np.mean(lats), np.mean(lons)]
    zoom_level = 10 + zoom_bias

    # Create map
    m = folium.Map(location=center, zoom_start=zoom_level,
                   tiles="http://mt1.google.com/vt/lyrs=y&z={z}&x={x}&y={y}",
                   attr="Google")

    # Add each polygon to the map
    for poly_coords in polygons:
        lat_lon_poly = [(y, x) for x, y in poly_coords]
        folium.Polygon(locations=lat_lon_poly, color="blue", fill=True, fill_opacity=0.4).add_to(m)

    # Add lat-lon popup
    folium.LatLngPopup().add_to(m)

    return m

def load_polygons_from_database(db_table, db_columns=[], db_geom_column = 'geom',
                                db_output_crs = 'EPSG:4283', DB_SCHEMA = "public") -> gpd.GeoDataFrame:
    """
    Load polygons directly from the configured PostgreSQL/PostGIS table.
    """
    
    try:
        import psycopg2  # type: ignore
        from psycopg2 import sql  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Database polygon loading requires the 'psycopg2-binary' package. "
            "Install with: pip install psycopg2-binary"
        ) from exc

    conn = psycopg2.connect(
        host="db-aurora-dea-fire-severity.cluster-cxhoeczwhtar.ap-southeast-2.rds.amazonaws.com",
        dbname="fire_severity_product",
        port="5432",
        user="processing_user_ro",
        password="isrTK76q\=1=!11XE^")

    table_identifier = (
        sql.Identifier(DB_SCHEMA, db_table)
        if DB_SCHEMA
        else sql.Identifier(db_table)
    )
    select_clause = sql.SQL(", ").join(sql.Identifier(col) for col in db_columns)
    query = sql.SQL(
        "SELECT {fields}, ST_AsGeoJSON({geom}) AS geom_geojson FROM {table}"
    ).format(
        fields=select_clause,
        geom=sql.Identifier(db_geom_column),
        table=table_identifier,
    )

    print(
        f"Querying polygons from table '{DB_SCHEMA + '.' if DB_SCHEMA else ''}{db_table}'..."
    )
    records: list[dict[str, Any]] = []
    failures = 0

    try:
        with conn, conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            print(f"Retrieved {len(rows)} rows from database.")
            for idx, row in enumerate(rows):
                attr_values = row[:-1]
                geom_raw = row[-1]
                try:
                    json_start = geom_raw.find("{")
                    geom_str = geom_raw[json_start:] if json_start >= 0 else geom_raw
                    geom_mapping = json.loads(geom_str)
                    geom_obj = shape(geom_mapping)
                except Exception as exc:
                    failures += 1
                    print(f"Warning: Failed to parse geometry for row {idx}: {exc}")
                    continue

                record = {col: val for col, val in zip(db_columns, attr_values)}
                record["geometry"] = geom_obj
                records.append(record)
    finally:
        conn.close()

    if failures:
        print(f"Skipped {failures} rows due to geometry parsing errors.")
    if not records:
        return gpd.GeoDataFrame(
            columns=[*db_columns, "geometry"], crs=db_output_crs
        )

    return gpd.GeoDataFrame(records, crs=db_output_crs)


def perform_spatial_dissolve(poly, id_list) -> gpd.GeoDataFrame:
    """takes a list of ids and the corrosponding geopandas dataframe and checks for spatial overlap 
    to see if polygons are actually the same fire or not"""

    geom_to_process = []

    
    for fire in id_list:
        subset = poly[poly['fire_id'] == fire].copy()
        
        # Find largest
        area_list = subset['area_ha'].unique()
        largest = subset[subset['area_ha'] == area_list.max()]
        largest_dc = largest.geometry.iloc[0]
        
         # Check overlaps
        do_they_overlap = subset.geometry.intersects(largest_dc)
        
        # Non-overlapping shapes
        unique_shapes = subset[~do_they_overlap]
        for idx, row in unique_shapes.iterrows():
            geom_to_process.append(gpd.GeoDataFrame(row.to_frame().T, geometry='geometry', crs=poly.crs))  # Convert Series to DataFrame
        
        # Overlapping shapes dissolved
        non_unique_shapes = subset[do_they_overlap]
        
        agg = {col: "first" for col in subset.columns if col not in ["geometry", "ignition_date", "capt_date", "date_processed"]}
        agg.update({"ignition_date": "min", "capt_date": "min", "date_processed": "max"})
        
        combine_to_one = non_unique_shapes.dissolve(aggfunc=agg)
        geom_to_process.append(combine_to_one)
    return gpd.GeoDataFrame(pd.concat(geom_to_process, ignore_index=True), crs=poly.crs)

def map_burn_severity(random_fire):

    print('CONDUCTING BURN SEVERITY MAPPING', random_fire)
        
    gpgon = datacube.utils.geometry.Geometry(random_fire.iloc[0].geometry, crs=random_fire.crs)
    
    fire_date = process_date(random_fire.ignition_date.iloc[0])

    if fire_date == None:
        
        #try from capture date
        fire_date = process_date(random_fire.capt_date.iloc[0])

        if fire_date == None:
            print("EMPTY Ignition DATE")
            return
        else:
            fire_date = datetime.strftime(
                ((datetime.strptime(fire_date, "%Y-%m-%dT%H:%M:%S.%fZ")) - timedelta(days=30)),
                "%Y-%m-%dT%H:%M:%S.%fZ")
    
    try:
        assumed_extinguish_date = process_date(random_fire.date_processed.iloc[0])


        
    except AttributeError as e:
        
        print('this fire has no processed date :(')
        return
    
    landcover_year = str(int(fire_date[0:4]) - 1)


    
    #Harcoding in WA's buffers, pre-immage witin 50 days before, post fire is 60 days after
    
    # Calculate the start and end date for baseline data load
    start_date_pre = datetime.strftime(
        ((datetime.strptime(fire_date, "%Y-%m-%dT%H:%M:%S.%fZ")) - timedelta(days=50)), '%Y-%m-%d')
    end_date_pre = datetime.strftime(
        ((datetime.strptime(fire_date, "%Y-%m-%dT%H:%M:%S.%fZ")) - timedelta(days=1)),
        '%Y-%m-%d')
    
       
    # Calculate start end date for post fire data load

        #if not extinguish date given assume 15 days
    extinguish_date = datetime.strftime(
        ((datetime.strptime(assumed_extinguish_date, "%Y-%m-%dT%H:%M:%S.%fZ")) - timedelta(days=7)),
        "%Y-%m-%dT%H:%M:%S.%fZ")

        #start date for post fire data load is extinguish date
    start_date_post = datetime.strftime(
        (datetime.strptime(extinguish_date, "%Y-%m-%dT%H:%M:%S.%fZ")),
        '%Y-%m-%d')
    

    end_date_post = datetime.strftime(
        ((datetime.strptime(start_date_post, "%Y-%m-%d")) + timedelta(days=60)),
        '%Y-%m-%d')

    
    baseline = load_ard(dc=dc,
                            products=['ga_s2am_ard_3', 'ga_s2bm_ard_3','ga_s2cm_ard_3'],
                            geopolygon=gpgon,
                            cloud_mask = 's2cloudless',
                            # x=study_area_lon,
                            # y=study_area_lat,
                            time=(start_date_pre, end_date_pre),
                            measurements=measurements,
                            min_gooddata=0.99, #we want no clouds for the pre-fire image
                            output_crs=output_crs,
                            resolution=resolution,
                            group_by='solar_day',
                            dask_chunks= {"x": 2048, "y": 2048})
    
    if baseline.time.size == 0:
        #try load again with less strict controls on clouds
        baseline = load_ard(dc=dc,
                            products=['ga_s2am_ard_3', 'ga_s2bm_ard_3','ga_s2cm_ard_3'],
                            geopolygon=gpgon,
                            cloud_mask = 's2cloudless',
                            # x=study_area_lon,
                            # y=study_area_lat,
                            time=(start_date_pre, end_date_pre),
                            measurements=measurements,
                            min_gooddata=0.50, #we want no clouds for the pre-fire image
                            output_crs=output_crs,
                            resolution=resolution,
                            group_by='solar_day',
                            dask_chunks= {"x": 2048, "y": 2048})

        #find most recent clean pixel :)
        closest_Bl = find_latest_valid_pixel(baseline)
    
    else:
    # (-1 is the most recent image from the stack)
        closest_Bl = baseline.isel(time=-1)
    
    debug_layer_blank = xr.ones_like(closest_Bl.nbart_red)
    
    #Calculate NBR
    pre_nbr = calculate_indices(closest_Bl, 
                                 index='NBR', 
                                 collection='ga_s2_3', 
                                 drop=True)

    # Load post-fire data from all Sentinel 2 satellites
    post = load_ard(dc=dc,
                    products=['ga_s2am_ard_3', 'ga_s2bm_ard_3','ga_s2cm_ard_3'],
                    geopolygon=gpgon,
                    cloud_mask = 's2cloudless',
                    # x=study_area_lon,
                    # y=study_area_lat,
                    time=(start_date_post, end_date_post),
                    min_gooddata=0.50,
                    measurements=measurements,
                    output_crs=output_crs,
                    resolution=resolution,
                    group_by='solar_day',
                    dask_chunks= {"x": 2048, "y": 2048})
    
    if post.time.size == 0:
        
            post = load_ard(dc=dc,
                    products=['ga_s2am_ard_3', 'ga_s2bm_ard_3','ga_s2cm_ard_3'],
                    geopolygon=gpgon,
                    cloud_mask = 's2cloudless',
                    # x=study_area_lon,
                    # y=study_area_lat,
                    time=(start_date_post, end_date_post),
                    min_gooddata=0.10,
                    measurements=measurements,
                    output_crs=output_crs,
                    resolution=resolution,
                    group_by='solar_day',
                    dask_chunks= {"x": 2048, "y": 2048})
            if post.time.size == 0:
                print('no post-fire data avalible', random_fire.fire_id)
                return
    
    #calculate NBR for post fire
    post_nbr = calculate_indices(post, 
                                 index='NBR', 
                                 collection='ga_s2_3', 
                                 drop=True)
    
    #calculate MINIMUM nbr in time stack
    max_nbr = post_nbr.min('time')

    #calculate the delta NBR

    deltaNBR =  pre_nbr - max_nbr

    landcover = dc.load(
                    product='ga_ls_landcover_class_cyear_3',
                    geopolygon=gpgon,
                    time=(landcover_year),
                    output_crs=output_crs,
                    resolution=resolution,
                    group_by='solar_day',
                    dask_chunks= {"x": 2048, "y": 2048})
    
    landcover = landcover.isel(time=0)
    
    #remove time dimention
    
    grass_class = [3, 14, 15, 16, 17, 18, 21, 32, 33, 34, 35, 36, 39, 50, 51, 52, 
                   53, 54, 57, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90,
                   91, 92, 94, 95, 96, 97]
    #these are the land Cover classes that are grass dominated
    
    grass_mask = xr.zeros_like(landcover.level4)
    
    for x in grass_class:
        x_pixels = (landcover.level4 == x ) * 1
    
        grass_mask = grass_mask + x_pixels
    # apply thesholds to get severity classes
    #the different severity classes are labled by stacking the extents together
    
    low = (deltaNBR >= 0.1) * 2
    medium = (deltaNBR >= 0.27) * 1
    high = (deltaNBR >= 0.44) * 1
    Very_high = (deltaNBR >= 0.66) * 1
    
    severity_woody = (low + medium + high + Very_high) #combine the woody severity classes
    severity_woody = (severity_woody.NBR.where(grass_mask== 0, 0))
    #mask woody burn severity by grass extent
    
    severity_grass = grass_mask.where(deltaNBR.NBR >= 0.1, 0)
    #this gives us where both grass and burnt are true = grass burn extent.
    
    severity = severity_woody + severity_grass
    #combine woody and grass burn classes

    #calculate NBR for post fire
    post_MNDWI = calculate_indices(post, 
                                 index='MNDWI', 
                                 collection='ga_s2_3', 
                                 drop=True)
    
    max_MNDWI = post_MNDWI.max('time')
    # max_MNDWI.MNDWI.plot()
    
    post_water = debug_layer_blank.where(max_MNDWI.MNDWI > 0, 0) #define water extent
    #  #find the wettest observation per pixel
    
    #generate de-bug layer (and mask ^ result if requiered)
    # mask for pre-fire cloud and and add that to de-bug
    
    new_debug = post_water + (debug_layer_blank.where(closest_Bl.oa_s2cloudless_mask == 2, 0))*10 #add cloud to debug 
    
    #mask for persistant post-fire cloud
    #first remove contiguity from cloudmask
    post_cloud = post.oa_s2cloudless_mask.where(post.oa_s2cloudless_mask >= 1, 1)
    persistant_cloud = post_cloud.min('time') #find any pixels that have had no clear observation accross stack
    
    new_debug = new_debug + (debug_layer_blank.where(persistant_cloud == 2, 0))*100 #add post-fire cloud to debug
    
    #now contiguity
    #pre-fire contiguity
    
    new_debug = new_debug + (debug_layer_blank.where(closest_Bl.oa_nbart_contiguity != 1, 0))*1000
    
    #post-fire contiguity
    post_contiguity = post.oa_nbart_contiguity.where(post.oa_nbart_contiguity == 1, 0) #turns all non 1 values to 0
    persistant_cont = post_contiguity.max('time') #find maximum contiguity value over stack
    
    new_debug = new_debug + (debug_layer_blank.where(persistant_cont != 1, 0))*10000 # add post fire contiguity to de-bug

    # #save_debug layer to file too
    # write_cog(new_debug.compute(), f'/home/jovyan/gdata1/projects/Hazards/burn_severity/results/debug_file/DEA_burn_severity_debug_{fire_id_forsave}.tif',
    #          overwrite=True)

    
    
    severity = severity.where(new_debug ==0, 6) #ensure dataclasses from de-bug are no data 

    severity = severity.astype('int32')
    # convert to vectors
    severity_vectors = xr_vectorize(severity, attribute_col='index',  crs=output_crs)
    
    #reproject fire boundary to AUS albers
    fire_boundary = random_fire.to_crs(output_crs)
    
    #clip to polygon
    cliped_severity_vectors = severity_vectors.clip(fire_boundary)
    
    #dissolve to join all polygons with same severity classes
    aggrigated_severity = cliped_severity_vectors.dissolve(by='index')

        #test if geometry is empty
    if aggrigated_severity.empty == True:
        print('she empty')
        return

    # give output polygons the extent polygon's attributes
    
    for att in attribute_list:
        if att in date_atts:
            aggrigated_severity[att] = process_date(random_fire[att].iloc[0])
        else: 
            aggrigated_severity[att] = random_fire[att].iloc[0]
    
    #calculate and add area 
    aggrigated_severity['area_ha'] = round((aggrigated_severity['geometry'].area)/10000, 2)
    
    #calculate and add perimiter (length measures perimiter of a multipoly)
    aggrigated_severity['perim_km'] =round((aggrigated_severity['geometry'].length)/1000, 2)

    #add severity lable attribues as desiered 
    aggrigated_severity['severity_rating'] = aggrigated_severity.index.astype(int)
    aggrigated_severity['severity_class'] = aggrigated_severity['severity_rating'].map(severity_list_dict)

    #add our assumed extinguish date to shapefile
    aggrigated_severity['extinguish_date'] = extinguish_date

    return aggrigated_severity, new_debug
