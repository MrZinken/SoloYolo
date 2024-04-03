import numpy as np
from PIL import Image
from ultralytics import YOLO
from osgeo import gdal, ogr, osr
import cv2
import os



def convert_to_resized_jpg(tiff_path):
    # convert tif to jpg
    tiff_img = Image.open(tiff_path)
    # Convert the image to JPG and PNG
    jpg_img = tiff_img.convert("RGB")
    resized_jpg_image = jpg_img.resize((9600, 9600), resample=Image.BOX)
    # jpg_path = os.path.join(input_folder, base_filename + ".jpg")
    # finale_image.save(jpg_path)

    return resized_jpg_image


def split_image(image):
    # Iterate through the image and save each piece
    for row in range(num_rows):
        for col in range(num_cols):
            left = col * tile_size
            upper = row * tile_size
            right = left + tile_size
            lower = upper + tile_size

            # Crop the piece
            piece = image.crop((left, upper, right, lower))

            # Save the piece with filename as part of the name
            piece_filename = f"piece_{row}_{col}.jpg"
            piece_path = os.path.join(tile_folder, piece_filename)
            piece.save(piece_path)


def predict(image, object_class, conf=0.5):
    # Predict with the model
    results = model(image, conf, classes = object_class)  # predict on an image
    # Save the results with the same filename as the input
    base_filename = os.path.splitext(filename)[0]
    for i, result in enumerate(results):
        # boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        # keypoints = result.keypoints  # Keypoints object for pose outputs
        # probs = result.probs  # Probs object for classification outputs
        # print('Number of objects in the picture: ', len(masks))     # number of detected objects
        output_path = os.path.join(tile_folder, f'{base_filename}.png')

        if (masks is None):
            # Create a new black image
            black_image = Image.new("RGB", (640, 640), color="black")
            black_image.save(output_path)
        else:
            mask = create_mask(masks)
            mask.save(output_path)  # Save the composite mask image
            # area = functions.calculate_area(mask)



def create_mask(masks):
    # Assuming `masks` is a list containing segmentation masks obtained from the YOLO model
    # Create an empty composite mask array
    composite_mask = np.zeros_like(masks[0].data[0].cpu().numpy(), dtype=np.uint8)

    # Assign different intensity values to each mask and overlay them onto the composite mask
    for i, mask in enumerate(masks, start=1):
        # Assign a unique intensity value (255/i) to each mask
        mask_data = (mask.data[0].cpu().numpy() * (255 // i)).astype(np.uint8)
        # Add the mask to the composite mask
        composite_mask += mask_data

    # Create an image from the composite mask array
    mask_img = Image.fromarray(composite_mask, "L")

    # Binarize the mask (thresholding)
    threshold = 2  # Adjust the threshold value as needed
    binary_mask = mask_img.point(lambda x: 0 if x < threshold else 255, '1')

    # return the composite mask image
    return binary_mask


def merge_images_resize(tile_folder):
    # Create a new blank image to merge pieces onto
    merged_image = Image.new("RGB", (num_cols * 640, num_rows * 640))

    # Iterate through the pieces and paste them onto the merged image
    for row in range(num_rows):
        for col in range(num_cols):
            piece_filename = f"piece_{row}_{col}.png"
            piece_path = os.path.join(tile_folder, piece_filename)
            if os.path.exists(piece_path):
                piece = Image.open(piece_path)
                merged_image.paste(piece, (col * 640, row * 640))
                os.remove(piece_path)
                # os.remove(piece_path)
            else:
                print(f"Piece {piece_filename} not found.")

    merged_image = merged_image.resize((10000, 10000), resample=Image.BOX)
    #output_path = os.path.join(output_folder, "fertig.png")
    #merged_image.save(output_path)
    return merged_image


def connect_close_masks(binary_mask, kernel_size):
    #connect masks, that belong to one entity, 
    #but are seperated because of slicing and detection errors
    
    # Convert Image to NumPy array
    binary_mask_np = np.array(binary_mask) 
    # Ensure binary mask is in uint8 format
    binary_mask_np = binary_mask_np.astype(np.uint8)

    # Convert RGB binary mask to grayscale
    grayscale_mask = cv2.cvtColor(binary_mask_np, cv2.COLOR_BGR2GRAY)

    # Apply morphological closing to fill small gaps between blocks
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    closed_mask = cv2.morphologyEx(grayscale_mask, cv2.MORPH_CLOSE, kernel)

    # Find contours in the closed mask
    contours, _ = cv2.findContours(
        closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create a blank mask for the massive block
    massive_block_mask = np.zeros_like(binary_mask)

    # Draw contours of all connected components onto the mask
    cv2.drawContours(massive_block_mask, contours, -1,
                     (255, 255, 255), thickness=cv2.FILLED)

    return massive_block_mask


def convert_png_to_tif(input_array):
    # Convert the NumPy array to a PIL Image

    pil_image = Image.fromarray(input_array)

    tif_path = os.path.join(output_folder, f'{base_filename}.tif')
    pil_image.save(tif_path)

    return tif_path


def raster_to_vector(tif_file, world_file, output_gpkg, target_srs=None):
    # Read TIFF mask and its world file to get georeferenced extent

    tif_ds = gdal.Open(tif_file)
    world_transform = np.loadtxt(world_file)
    pixel_width = world_transform[0]
    pixel_height = world_transform[3]
    top_left_x = world_transform[4]
    top_left_y = world_transform[5]


    # Convert binary mask to vector polygons
    src_band = tif_ds.GetRasterBand(1)
    mask_array = src_band.ReadAsArray()
    contours, _ = cv2.findContours(mask_array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create GeoPackage and layer
    driver = ogr.GetDriverByName("GPKG")
    out_ds = driver.CreateDataSource(output_gpkg)
    
    # Set the spatial reference system
    if target_srs is not None:
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(target_srs)
    else:
        # Use the same spatial reference system as the input raster
        srs = osr.SpatialReference()
        srs.ImportFromWkt(tif_ds.GetProjectionRef())

    layer = out_ds.CreateLayer("mask", srs, ogr.wkbPolygon)

    # Add field to store pixel value
    field_defn = ogr.FieldDefn("Value", ogr.OFTInteger)
    layer.CreateField(field_defn)

    # Create feature for each contour
    for contour in contours:
        if len(contour.squeeze().shape) != 2:  # Check if contour is not multidimensional
            continue  # Skip non-iterable contours
        ring = ogr.Geometry(ogr.wkbLinearRing)  # Create a new linear ring for each contour
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


def combine_geopackages(input_folder, output_geopackage):
    """
    Combines features from multiple GeoPackages into a single GeoPackage.

    Parameters:
        input_folder (str): Path to the folder containing input GeoPackage files.
        output_geopackage (str): Output GeoPackage filename.
    """
    # Create output GeoPackage
    driver = ogr.GetDriverByName("GPKG")
    out_ds = driver.CreateDataSource(output_geopackage)

    # Check if the output GeoPackage creation was successful
    if out_ds is None:
        print(f"Error: Failed to create output GeoPackage {output_geopackage}.")
        return

    # List all files in the input folder
    input_geopackages = [os.path.join(input_folder, file) for file in os.listdir(input_folder) if file.endswith('.gpkg')]

    # Track the current fid value
    current_fid = 0

    # Iterate over input GeoPackages
    for input_geopackage in input_geopackages:
        # Open input GeoPackage
        in_ds = ogr.Open(input_geopackage)
        if in_ds is None:
            print(f"Error: Could not open {input_geopackage}")
            continue

        # Iterate over layers in input GeoPackage
        for i in range(in_ds.GetLayerCount()):
            layer = in_ds.GetLayerByIndex(i)

            # Create corresponding layer in output GeoPackage if it doesn't exist
            if out_ds.GetLayerByName(layer.GetName()) is None:
                out_layer = out_ds.CreateLayer(layer.GetName(), layer.GetSpatialRef(), layer.GetGeomType())

                # Copy fields from input to output layer
                layer_defn = layer.GetLayerDefn()
                for j in range(layer_defn.GetFieldCount()):
                    field_defn = layer_defn.GetFieldDefn(j)
                    out_layer.CreateField(field_defn)

            else:
                # Get existing layer
                out_layer = out_ds.GetLayerByName(layer.GetName())

            # Copy features from input to output layer
            for feature in layer:
                # Assign a new fid value
                current_fid += 1
                feature.SetFID(current_fid)
                out_layer.CreateFeature(feature)
        # Close input GeoPackage
        in_ds = None
    # Close output GeoPackage
    out_ds = None


# load a custom model
model = YOLO('/home/kai/Documents/SoloYolo/runs/segment/train6/weights/best.pt')
# Input and output folders
input_folder = '/media/kai/Bonn/DOP_2022_2_5_cm_rgb'
output_folder = '/home/kai/Documents/bonn/outpput'
tile_folder = '/home/kai/Documents/bonn/tiles'


output_geopackage = '/home/kai/Desktop/solar_panel_new.gpkg'

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(tile_folder, exist_ok=True)

num_rows = 15
num_cols = 15
tile_size = 640

# 0 = solar panel
# 1 = solar thermie
# 2 = roof window
object_class = 0

#higher values only dectects objects with high confidence
confidence = 0.5

target_srs=25832
#specify the georeference 

start_at_name = "72752250.tif"  # Specify the starting point
found_start = True     # Set to True if you want to start at the beginning

# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    if not found_start:
        if filename.startswith(start_at_name):
            found_start = True
    elif filename.endswith('tif'):
    # Filter by supported image formats
        #convert to jpg with a size, that is divisible by 640
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        input_path = os.path.join(input_folder, filename)
        jpg_image = convert_to_resized_jpg(input_path)
        #split image into tiles wit 640 by 640
        split_image(jpg_image)
        for filename in os.listdir(tile_folder):
            tile_path = os.path.join(tile_folder, filename)
            input_image = Image.open(tile_path)
            #predict and create binary mask png
            predict(input_image, object_class, confidence)
            os.remove(tile_path)
        #merge image 
        merged_mask = merge_images_resize(tile_folder)
        #remove seperating lines between instance
        mask_wo_gaps = connect_close_masks(merged_mask, 20)
        #convert png to tif
        tif_path = convert_png_to_tif(mask_wo_gaps)
        #specify world file input - and geopackage output path
        world_file = os.path.join(input_folder, f'{base_filename}.tfw')
        output_gpkg = os.path.join(output_folder, f'{base_filename}.gpkg')
        raster_to_vector(tif_path, world_file, output_gpkg, target_srs=25832)
        os.remove(tif_path)

#combine_geopackages(output_folder, output_geopackage)
