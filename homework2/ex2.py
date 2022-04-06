import json
from PIL import Image, ImageDraw
import os

# Old code, not used. Ex 2 was done manually.

image_path = 'images_original/'
image_files = os.listdir(image_path)

label_path = 'labels/'
label_files = os.listdir(label_path)

labeled_images_path = 'labeled_images/'

def drawBrethingHoleRectangle(image):
    draw = ImageDraw.Draw(image)
    breathingHoleRectangle = [
        (breathingHole["left"], breathingHole["top"]), 
        (breathingHole["left"]+breathingHole["width"], breathingHole["top"]+breathingHole["height"])
    ]
    draw.rectangle(breathingHoleRectangle, outline ="red", width=5)

def drawBodyRectangle(image):
    draw = ImageDraw.Draw(image)
    bodyRectangle = [
        (body["left"], body["top"]), 
        (body["left"]+body["width"], body["top"]+body["height"])
    ] 
    draw.rectangle(bodyRectangle, outline ="blue", width=5) 

def parseCoordinates(data):
    annotatedResult = data['annotatedResult']
    inputImageProperties =  annotatedResult['inputImageProperties']
    boundingBoxes = annotatedResult['boundingBoxes']
    breathingHole = boundingBoxes[0]
    body = boundingBoxes[1]
    return breathingHole, body

# Draw labeling boxes on images
for i in range(0, len(label_files)):
    with open(label_path + label_files[i], 'r') as json_file:
        # Load coordinates
        breathingHole, body = parseCoordinates(json.load(json_file)[0])
        # Load images
        image = Image.open(image_path + image_files[i])
        # Crop body
        image_cropped = image.crop((body["left"], body["top"], 
            body["left"]+body["width"], body["top"]+body["height"]))
        image_cropped.save('templates/body/' + image_files[i])
        # Crop breathing hole
        image_cropped = image.crop((breathingHole["left"], breathingHole["top"], 
            breathingHole["left"]+breathingHole["width"], breathingHole["top"]+breathingHole["height"]))
        image_cropped.save('templates/breathingHole/' + image_files[i])
        # Draw boxes and save
        drawBrethingHoleRectangle(image)
        drawBodyRectangle(image)
        image.save(labeled_images_path + image_files[i])
