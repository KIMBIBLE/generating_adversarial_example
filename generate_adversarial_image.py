import sys
from operator import eq
import numpy as np
from keras.preprocessing import image
from keras.applications import inception_v3
from keras import backend as K
from PIL import Image

# parse image's name to predict
if int(len(sys.argv)) < 2:
    print("Usage {} : [input_image_name] [ImageNet Object Number]"\
          .format(sys.argv[0]))
    exit()

img_name = sys.argv[1]

# Choose an ImageNet object to fake
if int(len(sys.argv)) is not 3:
    print("[*] Set ImageNet Object to Starfish(327).")
    print("    If you want to set other object type,")
    print("    you have to excute this program like following.")
    print("    python {} [ImageNet Object Number]".format(sys.argv[0]))

    # default object class : 327 (starfish)
    object_type_to_fake = 327
    object_type_name = "starfish"
else:
    object_type_to_fake = int(sys.argv[2])

with open("./imagenet_classes_map.txt", "r") as f:
    lines = f.readlines()
    for line in lines:
        line_num = int(line.split(',')[0])
        line_name = line.split(',')[1][1:]
        if  line_num == object_type_to_fake:
            object_type_name = line_name.split('\n')[0]
            print("[*] Set ImageNet Object to {} : {}".format(object_type_name, object_type_to_fake))
            break

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
    # Grab a reference to the first and last layer of the neural net
    model_input_layer = model.layers[0].input
    model_output_layer = model.layers[-1].output


# Load the image file and convert it to a numpy array
try:
    print("[-] Loading images")
    img = image.load_img(img_name, target_size=(299, 299))
    original_image = image.img_to_array(img)

except Exception as e:
    print("[!] Fail to load image \"{}\"".format(img_name))
    print(e)
    exit()
else:
    print("[*] Loading images done")


# Scale the image so all pixel intensities are between [-1, 1] as the model expects
print("[*] Scaling image...")
original_image /= 255.
original_image -= 0.5
original_image *= 2.


# Add a 4th dimension for batch size (as Keras expects)
original_image = np.expand_dims(original_image, axis=0)


# Pre-calculate the maximum change we will allow to the image
# We'll make sure our hacked image never goes past this so it doesn't look funny.
# A larger number produces an image faster but risks more distortion.
max_change_above = original_image + 0.01
max_change_below = original_image - 0.01


# Create a copy of the input image to hack on
hacked_image = np.copy(original_image)


# How much to update the hacked image in each iteration
learning_rate = 0.1


# Define the cost function.
# Our 'cost' will be the likelihood out image is the target class according to the pre-trained model
print("[*] Defining the cost function")
cost_function = model_output_layer[0, object_type_to_fake]


# We'll ask Keras to calculate the gradient based on the input image and the currently predicted class
# In this case, referring to "model_input_layer" will give us back image we are hacking.
gradient_function = K.gradients(cost_function, model_input_layer)[0]


# Create a Keras function that we can call to calculate the current cost and gradient
grab_cost_and_gradients_from_model = K.function([model_input_layer, K.learning_phase()], [cost_function, gradient_function])


cost = 0.0


# In a loop, keep adjusting the hacked image slightly so that it tricks the model more and more
# until it gets to at least 80% confidence
print("[-] Generating adversarial image in loop...")
while cost < 0.80:
    # Check how close the image is to our target class and grab the gradients we
    # can use to push it one more step in that direction.
    # Note: It's really important to pass in '0' for the Keras learning mode here!
    # Keras layers behave differently in prediction vs. train modes!
    cost, gradients = grab_cost_and_gradients_from_model([hacked_image, 0])

    # Move the hacked image one step further towards fooling the model
    hacked_image += gradients * learning_rate

    # Ensure that the image doesn't ever change too much to either look funny or to become an invalid image
    hacked_image = np.clip(hacked_image, max_change_below, max_change_above)
    hacked_image = np.clip(hacked_image, -1.0, 1.0)
    print("[*] Model's predicted likelihood that the image is a {}: {:.8}%".format(object_type_name, cost * 100))


# De-scale the image's pixels from [-1, 1] back to the [0, 255] range
print("[*] De-scaling image")
img = hacked_image[0]
img /= 2.
img += 0.5
img *= 255.


# Save the hacked image!
print("[*] Saving generated image of adversarial example")
im = Image.fromarray(img.astype(np.uint8))
image_name_to_save = img_name.split('/')[-1]
im.save(image_name_to_save)
print("[*] All processing is done!")
print("    result will be located at ./results/{}".format(image_name_to_save))

