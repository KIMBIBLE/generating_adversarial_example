import sys
import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3

# parse image's name to predict
if int(len(sys.argv)) is not 2:
    print("Usage {} : [input_image_name]".format(sys.argv[0]))
    exit()

img_name = sys.argv[1]


# Load pre-trained image recognition model
try:
    print("[-] Loading model...")
    model = inception_v3.InceptionV3()

except Exception as e:
    print("[!] Fail to load module")
    print(e)
    exit()
else:
    print("[*] Loading model done.")


# Load the image file and convert it to a numpy array
try:
    print("[-] Loading images")
    img = image.load_img(img_name, target_size=(299, 299))
    input_image = image.img_to_array(img)

except Exception as e:
    print("[!] Fail to load image \"{}\"".format(img_name))
    print(e)
    exit()
else:
    print("[*] Loading images done")

# Scale the image so all pixel intensities are between [-1, 1] as the model expects
print("[*] Scaling image...")
input_image /= 255.
input_image -= 0.5
input_image *= 2.

# Add a 4th dimension for batch size (as Keras expects)
input_image = np.expand_dims(input_image, axis=0)


# Run the image through the neural network
try:
    print("[-] Start prediction")
    predictions = model.predict(input_image)
except Exception as e:
    print("[!] Error occured during prediction")
    print(e)
    exit()
else:
    print("[*] Success to prediction")

# Convert the predictions into text and print them
predicted_classes = inception_v3.decode_predictions(predictions, top=1)
imagenet_id, name, confidence = predicted_classes[0][0]
print("This is a {} with {:.4}% confidence!".format(name, confidence * 100))

