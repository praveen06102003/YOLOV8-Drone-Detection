**Setup YOLOv8 Model**

Load the YOLOv8 model pretrained on a dataset suitable for drone detection.
Configure the model for your specific detection task if needed.

**Load and Preprocess Image**

Read the image from your source (e.g., camera feed, file).
Preprocess the image to fit the input requirements of the YOLOv8 model (e.g., resizing, normalization).
**Run Inference**

Pass the preprocessed image to the YOLOv8 model to get predictions.
The model will output bounding boxes, class labels, and confidence scores for detected objects.

**Extract Coordinates**

Extract the bounding box coordinates (x, y, width, height) from the model's predictions.
Convert these coordinates into the format you need (e.g., specifying exact x and y coordinates).

**Display Results**

Draw bounding boxes around detected drones on the image.
Optionally, display or log the coordinates for further processing or analysis.
