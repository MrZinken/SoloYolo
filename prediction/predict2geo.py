import numpy as np
from PIL import Image
from ultralytics import YOLO
from osgeo import gdal, ogr, osr
from scipy.ndimage import binary_closing, label
import cv2
import os



def predict(image):
    # Predict with the model
    results = model(image, conf=0.4)  # predict on an image
    # Save the results with the same filename as the input
    base_filename = os.path.splitext(filename)[0]
    
    for i, result in enumerate(results):
        #boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        #keypoints = result.keypoints  # Keypoints object for pose outputs
        #probs = result.probs  # Probs object for classification outputs
        # print('Number of objects in the picture: ', len(masks))     # number of detected objects
        output_path = os.path.join(tile_folder, f'{base_filename}.npy')
        
        if masks is None:
            # Create a new black image (empty mask)
            empty_mask = np.zeros((640, 640), dtype=np.uint8)
            np.save(output_path, empty_mask)
        else:
            for j ,mask in enumerate(masks):
                mask_raw = results[0].masks[j].cpu().data.numpy().transpose(1, 2, 0)
            np.save(output_path, mask_raw) # Save the composite mask image

         


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
            

def merge_arrays(tile_folder, num_rows, num_cols):
    """
    Merge NumPy arrays from files in a folder into a single array.

    Parameters:
        tile_folder (str): Path to the folder containing the array files.
        num_rows (int): Number of rows in the grid of arrays.
        num_cols (int): Number of columns in the grid of arrays.

    Returns:
        numpy.ndarray: The merged array.
    """
    # Create a new blank array to merge pieces onto
    merged_array = None

    # Iterate through the pieces and paste them onto the merged array
    for row in range(num_rows):
        for col in range(num_cols):
            piece_filename = f"piece_{row}_{col}.npy"
            piece_path = os.path.join(tile_folder, piece_filename)
            if os.path.exists(piece_path):
                piece = np.load(piece_path)
                if merged_array is None:
                    merged_array = piece
                else:
                    merged_array = np.concatenate((merged_array, piece), axis=1)
            else:
                print(f"Piece {piece_filename} not found.")

    # Resize the merged array to the target size (10000x10000)
    if merged_array is not None:
        new_height = int(merged_array.shape[0] * 10000 / (num_rows * 640))
        new_width = int(merged_array.shape[1] * 10000 / (num_cols * 640))
        merged_array = cv2.resize(merged_array, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    return merged_array





def connect_close_masks(binary_mask, kernel_size):
    """
    Connects and closes gaps in a binary mask.

    Parameters:
        binary_mask (numpy.ndarray): The binary mask.
        kernel_size (int): The size of the structuring element for closing.

    Returns:
        numpy.ndarray: The connected and closed mask.
    """
    # Apply morphological closing to fill small gaps between blocks
    closed_mask = binary_closing(binary_mask, structure=np.ones((kernel_size, kernel_size)))

    # Label connected components
    labeled_mask, num_labels = label(closed_mask)

    # Create a blank mask for the massive block
    massive_block_mask = np.zeros_like(binary_mask)

    # Draw contours of all connected components onto the mask
    for i in range(1, num_labels + 1):
        massive_block_mask[labeled_mask == i] = 255

    return massive_block_mask.astype(np.uint8)


def convert_to_jpg(tiff_path):
    #convert tif to jpg
    tiff_img = Image.open(tiff_path)
    # Convert the image to JPG and PNG
    jpg_img = tiff_img.convert("RGB")
    jpg_image= jpg_img.resize((9600, 9600), resample=Image.BOX)
    #jpg_path = os.path.join(input_folder, base_filename + ".jpg")
    #finale_image.save(jpg_path)  

    return jpg_image

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

# load a custom model
model = YOLO('/home/kai/Documents/SoloYolo/runs/segment/best/best.pt')
# Input and output folders
input_folder = '/home/kai/Desktop/input'
output_folder = '/home/kai/Desktop/input/output'
tile_folder = '/home/kai/Desktop/input/tiles'

num_rows = 15
num_cols = 15
tile_size = 640
output_shape = (9600,9600)


# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(tile_folder, exist_ok=True)


# Iterate over the files in the input folder
for filename in os.listdir(input_folder):
    # Filter by supported image formats
    if filename.endswith(('.tiff', 'tif')):
        base_filename = os.path.splitext(os.path.basename(filename))[0]
        input_path = os.path.join(input_folder, filename)
        split_image(convert_to_jpg(input_path))
        for filename in os.listdir(tile_folder):
            tile_path = os.path.join(tile_folder, filename)
            input_image = Image.open(tile_path)
            predict(input_image)
        
        image = merge_arrays(tile_folder, 15, 15)   #remove seperating lines between masks
        np.save("/home/kai/Desktop/array.npy")
        finale_image = connect_close_masks(image, 20)
        output_path = os.path.join(output_folder, base_filename + ".npy") 
        #cv2.imwrite(output_path, finale_image)
        np.save(output_path, finale_image)
