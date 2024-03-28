# SoloYolo
## Framweork, das aus Überflug-Bildern der Stadt Bonn Objekte identifiziert und diese zu einem Geopackage zusammensetzt. In diesem Fall wurde das Netz auf Solarpanelee trainiert.

Im Folgenden soll der Weg von den Ausgangsbildern, über das Annotieren, Trainieren, Evaluieren hin zu dem fertigen Geopackage möglichst kleinschrittig erklärt werden, sodass auch Beginner im Bereich ML einen guten Einstieg finden. Etwas Verständnis vom Programmieren wird vorausgesetzt.


## Vorassetzung
Das Projekt wurde auf einem stärkeren Laptop entwickelt. Es wird jedoch empfohlen das Training und die Vorhersage von vielen Bildern auf einem Rechner mit dedizierter GPU laufen zu lassen, die CUDA unterstützt.
Die Überflugbilder der Stadt Bonn sind im TIF Format und haben eine Auflösung von 10000 x 10000 Pixel, während ein Pixel 2,5 x 2,5 cm Bodenfläche abdeckt. Da das Neuronale Netz 640 x 640 Pixel große JPG-Bilder bevorzugt, muss bei der Vorverarbeitung darauf geachtet werden, ob das Zerschneiden zu einem Bild mit den entsprechenden MAßen führt. Sonst muss der Code für die Vorverarbeitung und das spätere Zusammensetzen der Bilder angepasst werden. Strikte Voraussetzung für das Neuronale Netz sind quadratische Bilder mit einem Vielfachen von 32 Pixeln. 

Als IDE wurde VSCode unter Linux benutzt. Es wird empfohlen Anaconda3 als Virtuelles Environmet in Version 3.11.8 einzurichten. Eine Anleitung dazu findet man [hier](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). In diesem Environment muss dann noch die Bibliothek von Ultralytics installiert werden, die das Netz zur Segmentation, sowie erstuanlich simple Funktionen zum Training und Anwendung des Netzes bereitstellen. Dies erfolgt mit folgendem Befehl im Terminal:
```
# Install the ultralytics package using conda
conda install -c conda-forge ultralytics
```





















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

## Model

Semantic Segmentation Model by ultralytics: YOLOv8 m? l?
https://docs.ultralytics.com/de/tasks/segment/

## Testing

Different metrics to evaluate the models

## Design Choices

Roboflow is a handy tool to generate a dataset for instance segmentation. The instances can be annotadet with polygons and a tool called smart polygon, that sepperates the object automatically. It is not perfect though.

YOLOv8 by ultralytics is a popular model as the API is beginner friendly and training and deployment can be done on average ai stations.

## Deployment

?


