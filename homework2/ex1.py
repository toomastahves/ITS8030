from PIL import Image
import os

input_path = 'homework2/images_original/'
output_path = 'homework2/images_resized/'
images_original = os.listdir(input_path)

for i in range(0, len(images_original)):
    image = Image.open(input_path + images_original[i])
    new_image = image.resize((1024, 1024))
    new_image.save(output_path + images_original[i])
