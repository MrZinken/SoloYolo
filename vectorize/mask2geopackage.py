from osgeo import gdal, ogr, osr
import numpy as np
import cv2


def raster_to_vector(tif_file, world_file, output_gpkg):
    # Read TIFF mask and its world file to get georeferenced extent
    tif_ds = gdal.Open(tif_file)
    world_transform = np.loadtxt(world_file)
    pixel_width = world_transform[0]
    pixel_height = world_transform[3]
    top_left_x = world_transform[4]
    top_left_y = world_transform[5]
    width = tif_ds.RasterXSize
    height = tif_ds.RasterYSize
    bottom_right_x = top_left_x + width * pixel_width
    bottom_right_y = top_left_y + height * pixel_height
    geotransform = (top_left_x, pixel_width, 0, top_left_y, 0, pixel_height)

    # Convert binary mask to vector polygons
    src_band = tif_ds.GetRasterBand(1)
    mask_array = src_band.ReadAsArray()
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create GeoPackage and layer
    driver = ogr.GetDriverByName("GPKG")
    out_ds = driver.CreateDataSource(output_gpkg)
    srs = osr.SpatialReference()
    srs.ImportFromWkt(tif_ds.GetProjectionRef())
    layer = out_ds.CreateLayer("mask", srs, ogr.wkbPolygon)

    # Add field to store pixel value
    field_defn = ogr.FieldDefn("Value", ogr.OFTInteger)
    layer.CreateField(field_defn)

    # Create feature for each contour
    for contour in contours:
        ring = ogr.Geometry(ogr.wkbLinearRing)
        for point in contour.squeeze():
            x, y = point
            geo_x = top_left_x + x * pixel_width
            geo_y = top_left_y + y * pixel_height
            ring.AddPoint(geo_x, geo_y)
        ring.CloseRings()
        poly = ogr.Geometry(ogr.wkbPolygon)
        poly.AddGeometry(ring)
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetGeometry(poly)
        feature.SetField("Value", 1)  # Set pixel value field
        layer.CreateFeature(feature)
        feature = None
    
    # Save changes and close datasets
    out_ds = None
    tif_ds = None

# Example usage
png_file = "/home/kai/Desktop/62752000.tif"
world_file = "/home/kai/Desktop/62752000.tfw"
output_gpkg = "/home/kai/Desktop/62752000.gpkg"
raster_to_vector(png_file, world_file, output_gpkg)


#processing.run("gdal:polygonize", {'INPUT':'/home/kai/Desktop/62752000.png','BAND':1,'FIELD':'DN','EIGHT_CONNECTEDNESS':False,'EXTRA':'','OUTPUT':'TEMPORARY_OUTPUT'})