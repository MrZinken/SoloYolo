# SoloYolo
## Framweork, das aus Überflug-Bildern der Stadt Bonn Objekte identifiziert und diese zu einem Geopackage zusammensetzt. In diesem Fall wurde das Netz auf Solarpanelee trainiert.

![layer](images/mitlayer.png)

Im Folgenden soll der Weg von den Ausgangsbildern, über das Annotieren, Trainieren, Evaluieren hin zu dem fertigen Geopackage möglichst kleinschrittig erläutert werden, sodass auch Beginner im Bereich ML einen guten Einstieg finden. Etwas Verständnis vom Programmieren ist wünschenswert. Ich versuche alles so zu erklären, wie ich es mir als Einsteiger in die Welt des maschinellen Lernens gewünscht hätte. Bei Problemen, Anregungen und Ideen kann gerne ein Issue eröffnet werden. 

Der Bereich des maschinellen Lernens, der hier angewendet wird, heißt Semantische Segmentattion oder Instance Segmentation. Dabei wird ein Objekt in einem Bild erkannt und auf der Fläche des Objekts eine Maske erstellt. Dadurch lassen sich anschließend Analysen auf dieser Maske ausführen.


## Vorassetzung
Das Projekt wurde auf einem stärkeren Laptop entwickelt. Es wird jedoch empfohlen das Training und die Vorhersage von vielen Bildern auf einem Rechner mit dedizierter GPU laufen zu lassen, die CUDA unterstützt.
Die Überflugbilder der Stadt Bonn sind im TIF Format und haben eine Auflösung von 10000 x 10000 Pixel, während ein Pixel 2,5 x 2,5 cm Bodenfläche abdeckt. Da das Neuronale Netz 640 x 640 Pixel große JPG-Bilder bevorzugt, muss bei der Vorverarbeitung darauf geachtet werden, ob das Zerschneiden zu einem Bild mit den entsprechenden MAßen führt. Sonst muss der Code für die Vorverarbeitung und das spätere Zusammensetzen der Bilder angepasst werden. Strikte Voraussetzung für das Neuronale Netz sind quadratische Bilder mit einem Vielfachen von 32 Pixeln. 

Als IDE wurde VSCode unter Linux benutzt. Es wird empfohlen Anaconda3 als Virtuelles Environmet in Version 3.11.8 einzurichten. Eine Anleitung dazu findet man [hier](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html). In diesem Environment muss dann noch die Bibliothek von Ultralytics installiert werden, die das Netz zur Segmentation, sowie erstuanlich simple Funktionen zum Training und Anwendung des Netzes bereitstellen. Dies erfolgt mit folgendem Befehl im Terminal:
```
# Install the ultralytics package using conda
conda install -c conda-forge ultralytics
```
Sollte dies nicht funktionieren ist [hier](https://docs.ultralytics.com/quickstart/) ein Guide für die Einrichtung der Umgebung.

Falls die GPU Cuda-Treiber unterstützt, sollten diese noch installiert werden. Ein Tutorial dazu für Windows findet man [hier](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) und für Linux [hier](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).

Außerdem wird noch ein Account bei [Roboflow](https://roboflow.com/) benötigt. Hier kann man sehr intuitiv die Daten labeln und einen passenden Datensatz generieren. Dafür reicht eine kostenlose Mitgliedschaft.

Sollte die Einrichtung geklappt haben ist der schwerste Teil geschafft. Bei Problemen sind Google oder ChatGPT eine große Hilfe. 

## Aufbau des Projekts
Ein Projekt im Bereich des maschinellen Lernen teilen sich grundsätzlich auf in:
- Erstellen des Datensatzes
- Trainieren des Neuronalen Netzes
- Evaluieren der Performance
- Anwendung
Diese Schritte werden im Folgenden erläutert.

## Erstellen des Datensatzes
Die Qualität des Datensatzes ist absolut entscheidend für die Performance des Models. Aus minderwertigen Daten wird nie ein gut funktionierndes Model werden. Je größer und diversers der Datensatz ist umso besser kann das Model trainieren. Hier muss eine Abwägung zwischen Zeitaufwand, vorhanden Daten und Anforderungen getroffen werden. Dafür kann keine generelle Angabe getroffen werden, da dies auch von "Schwierigkeit" abhängt, die das Model bei der Bearbeitung der Daten hat. Außerdem besteht kein linearer Zusammenhang zwischen Größe des Datensatzes und der anschließenden Perfomrance. Dieser Zusammenhang wird eher durch beschränktes Wachstum definiert. In diesem Fall haben rund 1000 annotierte Bilder mit etwa 2200 Instanzen von Solarpanelen zu einer befriedigenden Performance geführt. Wenn die Ressourcen vorhandne sind führt ein Datensatz von 10000 Bilder aber zu einer höheren Perfomance und Robustheit.

### Ausgangspunkt und Ziel
Als Ausgangspunkt sind hier konkret Tiff Bilder mit einer Auflösung von 10000 x 10000 Pixeln gegeben und am Ende erhält man einen Datensatz von gelbelten/annotierten Bildern, die das zu detekierende Objekt, sowie typischen False Positives enthalten. False Positives sind Objekte die von dem Model erkannt werden, weil sie ähnliche Eigenschaften aufweisen, aber eben nicht der Klasse entsprechen, die man segmentieren möchte. Bei der Erkennung von Solarpaneleen sind das zum Beispiel Solarthermie Anlagen, Dachfenster und Überdachungen, aber auch blaue Autos oder Bahnschienen typische False Positives. 
Welche Objekte zu False Positives führen ist nicht direkt vorherzusagen. Darauf kann bei einem iterativen Vorgehen eingegangen werden indem die False Positives in den Datensatz aufgenommen werden. Um eine noch stärkere Abgrenzung zu erreichen, kann man für "starke" False Positives eine eigene Klasse erstellen, indem man sie extra labelt. Dadurch lernt das Model diese besser zu unterscheiden. Für Solarpanele wurden die Klassen Überdachung und Solarthermie verwendet. Das folgende Bild enthält kein Solarpanel und hilft dabei zu verstehen, vor welcher Herausforderung man steht, wenn man im urbanen Umfeld selektiv Solarpanele detektieren will.
![false positives](images/false_positives.jpg)

### Konkretes Vorgehen
Zunächst sollte man sich einen Querschnitt der verfügaberen Daten ansehen und ein Verständnis davon Etnwickeln wie Objekte, die man detektieren will aussehen. Die Varianz von simplen Objekten kann schon erstaunlich hoch sein und von Dingen abhängen, die man nicht antizipiert. Hat man eine Idee von den verfügbaren Daten und der Klasse von Objekten die man Segmentieren möchte, sollte man eine möglichst diverse Auswahl von Bildern zusammenstellen. Dabei ist eine hohe Varianz der Bilder entscheidend. Hat man diese Bilder gesammelt müssen diese zerschintten werden, damit sie dem Format entsprechen, das das Neuronale Netz verarbeiten kann. Die Funktion [slize.py](https://github.com/MrZinken/SoloYolo/tree/main/dataset) übernimmt dies. Hier müssen lediglich die Ordner spzifizert werden, indem man die zu zerschneidenden Bilder abgelegt hat und die "Schnipsel" abgelegt werden sollen. 
```
# Specify input and output folders
input_folder = "/home/kai/Desktop/2slice"
output_folder = "/home/kai/Desktop/sliced"
piece_size = 640  # Specify the size of each piece in pixels
```
Außerdem kann die Bildgoeße definiert werden, wobei 640x640 eine sinvoll ist. Sollten sich die Ausgansbilder unterscheiden, muss dieses Script angepasst werden. Im folgenden müssen immer wieder Pfade zu gewümschten Ordner spezifizert werden. Da die Variablen für die Pfade (hoffentlich) selbsterklärend sind, werde ich darauf nicht mehr genauer eingehen.

Lässt man das Skript laufen und hat die kleineren Bilder, muss man diese durchgehen und vor allem Bilder mit den gewünschten Klassen sammeln. Dies ist relativ aufwändig und erfordert viel Zeit und Konzentration. Es ist wichtig auch Objekte zu erkennen, die nur an den Rändern in das Bild rein reichen, damit diese auch später erkannt werden. Es ist wichtig hier auch typische False Positives mit in den Datensatz aufzunehmen und auch typische Background Bilder zu behalten. Dabei wird meistens ein Verhältnis von 10 zu 1 vorgeschlagen, von Bildern mit der Klasse zu Background Bildern. In diesem Fall wurde aber ein deutlich kleineres Verhältnis gewählt, da die False Positives ein großes Problem waren und das Model eher lernen musste, was es nicht markieren soll. 
Hat man seinen Vorverabeitetne Datensatz nun gespeichert, ist es sinvoll einen Backup davon zu erstellen, da jetzt schon viel Arbeit darin steckt.
Nun wechselt man zu Roboflow und erstellt ein Projekt:
![projekt_erstellen](images/projekt_erstellen.png)
Dann legt man den Namen des Projekts, die License und die Klassen fest, die man anntoieren will. Ganz wichtig an dieser Stelle ist, dass man unten die art des Projektes festlegt, in unserem Fall Instance Segmentation:
![einrichtung_projekt](images/einrichtung_projekt.png)
In die nun erscheinende Seite kann man per Drag and Drop die Bilder einfügen und diese hochladen. Mit klicken auf "Save and Continue", gelangt man auf eine Seite auf der man sich rechts zwischen 3 Methoden zum laben der Daten entscheiden kann. Hier sollte man "Start Manual Labeling" auswählen, da dies kostenlos ist und man die Kontrolle über die Qualität der Annotationen hat. In einem Untermenü kann ausgewählt werden, welchem Teammitglied, die Annotation zuordnen will. Wenn man Pech hat, muss man es sich selber zuordnen, indem man unten "Assign Images" anklickt. Auf der folgenden Seite gelangt man mit einem Klick auf Start Annotating endlich zu der Umgebung wo man mit dem Markieren der Daten beginnen kann.  
![annotieren](images/annotieren.png)
Das Vorgehen beim Annotieren, sollte intuitiv sein und hängt von den Objekten ab, die man markieren will. Dafür stehen einem grundsätzlich zwei Tools zur verfügung. Für das Labeln von Solarpanelen hat es sich angeboten das Polygon Tool zu verwenden, das rechts in dem obersten roten Kästchen liegt. Damit kann man Punkte definieren, die das Obejkt einschließen sollen. Es ist wichtig die Objekte Konsequent bis zum Rand zu markieren und darauf zu achten, dass diese nicht abgeschnitten werden. Alle Fehler die man hier begeht, werden von dem Model übernommen. Außerdem gibt es ein Tool zur automatischen Abgrenzung, genannt "Smart Polygon, das direkt darunter liegt. Dieses ist erfahungsgemäß aber nicht präzise genug und kostet durch die Korrektur mehr Zeit als die manuelle Abgrenzung. 
Das kann aber für Objekte die keine geraden Kanten haben und mehr Kontrast gegenüber dem Hintergrund haben anders sein. 
Ist das Obejkt abgegrenzt wird durch drücken der Eingabetaste bestätigt und anschließend kann die gewünschte Klasse oben links gewählt werden. 
Ist kein zu markierendes Obejkt in dem Bild vorhanden kann rechts unten das Bild als Background markiert werden(unterstes rotes Kästchen).
Ich rate dringend zu einem spannenden Hörbuch, damit diese Tätigkeit nicht zu monoton wird.

Sobald alle Bilder gelabelt wurde, können diese in der Projecktübersicht unter Annotate rechts oben mittels Klick auf "Add n images to Dataset" dem Datensatz hinzugefügt werden. Dabei kann die Aufteilung des Datensatzes bestimmt werden. Normalerweise werden 70% der Bilder für das Training des Models gewählt, während 20% für Validierung und 10% zum Testen verwendet werden. Diese Aufteilung kann so gelassen werden oder in speziellen Fällen variert werden. 
Die annotierten Bilder könne dann unter dem Reiter "Dataset" und "Health Chek" neben anderen Informationen eingesehen werden.
Unter "Generate" auf der linken Seite wird dann der Datensatz erstellt, der für unser Model zu verarbeiten ist. Beim "Preprocessing" sind keine Änderungen nötig, man kann direkt mit "Continue" fortfahren. 
Unter "Augmentation" findet man eine der hilfreichsten Tools von Roboflow. Bei der Datenaugmentation werden die gelabelten Bilder auf verschiedene Weisen verändert, während die Markierungen erhalten bleiben. Dadurch erhält man "gratis" einen größeren Datensatz, während das Model zuätzlich robuster gegenüber neuen Daten wird. Mit einem Klick auf "Add Augmentation Step" werden einem folgende Schritte angeboten:
![Augmentation](images/Augmentation.png)
Bedenkenlos kann hier Flip, Rotate angewendet werden. Außerdem sind leichte Veränderungen mittels Hue, Saturation, Brightness und Exposure sinnvoll. Wichtig ist, dass bei diesen Schritten nur Bilder entstehen, die auch wirklich unter normalen Bedingungen entstehen. Ansonsten wird das Model auf Bilder trainiert, die es niemals sehen wird. Die Parameter mit denen man die einzelnen Augmentation Schritte ausführt, sollten entsprechend konservativ gewählt werden. 
Anschließend dann mittels Klick auf "Continue" und "Generate" der Datensatz erstellt. Dieser ist unter Versions zu finden.
Um den Datensatz herunterladen zu können klickt man rechts oben auf "Export Dataset".
In dem Fenster das erscheint wählt man unter dem Reiter Format "Yolov8" und wählt das Kästchen "download zip to computer" und klickt auf "Continoue":
![Export](images/export.png)
Dann beginnt der Download des Datensatzes in dem benötigten Format.




