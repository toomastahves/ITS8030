from PIL import Image
import os

# Resize for neural network training
input_path = 'images/templates/'
images_original = os.listdir(input_path)

for i in range(0, len(images_original)):
    print(input_path + images_original[i])
    image_name = images_original[i].split('.')[0]
    
    # Load image
    image = Image.open(input_path + images_original[i])
    
    # Create image and copy transparent image there, transparent images will have same background color
    new_image = Image.new("RGBA", image.size, "BLACK")
    new_image.paste(image, (0, 0), image) 

    # Resizes, but also stretches
    new_image = image.resize((1024, 1024))
    
    # Convert to RGB image, loses transparency
    new_image = new_image.convert('RGB')

    new_image.save('images/train/{}.png'.format(image_name))
