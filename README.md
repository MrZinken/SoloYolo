# SoloYolo: Solar Panel Detection from Aerial Images

## Goal

Create a userfriendly workflow to detect, visualize and calculate solar panels on aerial images with the help of machine learinng/sematic segmentation. This task is done for the city of Bonn while doing my Praxisprojekt.

## Side Goals

learn a lot

## General Proceedure

After evaluating two projects, that have the same goal, I decided, that I train my own model with the data that has the same resolution of the data that should be analyzed. 

https://github.com/Kleebaue/multi-resolution-pv-system-segmentation

https://github.com/fvergaracontesse/hyperion_solar_net/blob/main/models/README.md

The tiff files provided have a size of 10.000x10.000 pixels and provide four channels(cir = rgb + near infrared). Each pixel covers an area of 2.5cm square. These images needs to be converted to jpg and split into tiles with the size of 640x640. 
These images are annotated in Roboflow with two classes: solar panels that generate electric energy and solarthermal system that heat up water. 
As they look similar and often appear in the same context I diceded to add them as a second class to avoid false positives. The city of Bonn is espacially intersted in pv systems that generate electric energy.

The predicted segmentation mask is resized to the original size of the original image. A vector image is generated and the area and number of the found instances will be calculated.

The results can then be used to apply geonalytical methods. 

## Generating a Segmentation Dataset for Solar Panels using Roboflow

### Overview:
This repository provides a step-by-step guide to generating a segmentation dataset for solar panels using Roboflow. The dataset creation process involves collecting images, annotating them, preprocessing the data, and training a segmentation model.

### Procedure:
1. **Collect Diverse Images**: Gather images containing solar panels from various sources, ensuring diversity in lighting conditions, angles, and environments.

2. **Annotate Images**: Use annotation tools like LabelImg or Roboflow Annotate to mark the locations of solar panels in each image, creating segmentation masks.

3. **Preprocess Data**: Resize images and convert annotations to a suitable format like COCO JSON or Pascal VOC XML.

4. **Upload to Roboflow**: Create a new dataset on the Roboflow platform and upload your annotated images and masks.

5. **Apply Data Augmentation**: Increase dataset diversity by applying transformations like rotation, flipping, and brightness adjustments.

6. **Generate Dataset Versions**: Experiment with different augmentation settings to create multiple dataset versions.

7. **Export Dataset**: Export the augmented dataset from Roboflow in a format compatible with your deep learning framework.

8. **Train Segmentation Model**: Train a segmentation model using your preferred deep learning framework (e.g., TensorFlow, PyTorch) and an appropriate architecture (e.g., U-Net, DeepLabv3).

9. **Evaluate Model Performance**: Assess the trained model's performance using metrics like Intersection over Union (IoU) and accuracy.

10. **Iterate and Improve**: Analyze the model's performance and iterate on the dataset and training process to enhance accuracy and robustness.




## Model

Semantic Segmentation Model by ultralytics: YOLOv8 m? l?
https://docs.ultralytics.com/de/tasks/segment/

## Testing

Different metrics to evaluate the diffirent models

## Design Choices

Roboflow is a handy tool to generate a dataset for instance segmentation. The instances can be annotadet with polygons and a tool called smart polygon, that sepperates the object automatically. It is not perfect though.

YOLOv8 by ultralytics is a popular model as the API is beginner friendly and training and deployment can be done on average ai stations.

## Deployment

?


