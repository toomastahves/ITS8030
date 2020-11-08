import json
from PIL import Image, ImageDraw
import os

image_path = 'homework2/images_original/'
image_files = os.listdir(image_path)

label_path = 'homework2/labels/'
label_files = os.listdir(label_path)

labeled_images_path = 'homework2/labeled_images/'

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

for i in range(0, len(label_files)):
    with open(label_path + label_files[i], 'r') as json_file:
        breathingHole, body = parseCoordinates(json.load(json_file)[0])
        image = Image.open(image_path + image_files[i])
        drawBrethingHoleRectangle(image)
        drawBodyRectangle(image)
        image.save(labeled_images_path + image_files[i])
