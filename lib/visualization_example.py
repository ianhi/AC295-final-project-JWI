from Segmentation import segmentation
import tensorflow as tf
from tensorflow.keras.models import Model
import keras.backend as K
import segmentation_models as sm
import numpy as np
import matplotlib.pyplot as plt

sm.set_framework('tf.keras')


DATAPATH = "data/image_and_masks/"
VISUALIZATONPATH = "visualizations/"

#-------------------------------------------------
# Train Segmentation Model
#-------------------------------------------------

# Training Mobilenet Pretrained
params={'backbone':'mobilenet',
        'loss': [
            tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            sm.losses.DiceLoss(class_weights=np.array([0.5,1,2]))
        ],
        'weights':[1,2],
        'augmentation':'yes',
        'weights_pretrained':'imagenet',
        'batch_size':10,
        'steps_per_epoch':10,
        'n_epochs':1,
        'encoder_freeze':'No'}

seg=segmentation(DATAPATH, ['background','cell1','cell2'], params=params)


# The following code for generating visualizations is from the following repo:
# https://towardsdatascience.com/understanding-your-convolution-network-with-visualizations-a4883441533b
# https://github.com/anktplwl91/visualizing_convnets
# Author: Ankit Paliwal

#-------------------------------------------------
# Feature Maps
#-------------------------------------------------

# Visualize layer activations (aka feature maps) for specified
# layers and activations
def visualize_layer_activations(seg, layer_names, activ_list, images_per_row=16):
  for layer_name, layer_activation in zip(layer_names, activ_list):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0, :, :, col * images_per_row + row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size, row * size : (row + 1) * size] = channel_image

    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.imshow(display_grid, aspect='auto', cmap='plasma')
    plt.savefig(VISUALIZATONPATH + layer_name + "_grid.jpg", bbox_inches='tight')

# Get model activations for sample image
def get_model_activations(seg):

    # Build model for layer outputs
    layer_outputs = [layer.output for layer in seg.model.layers]
    activation_model = Model(inputs=seg.model.input, outputs=layer_outputs)

    # Get sample image
    image, mask=next(seg.train_generator)
    sample_image, sample_mask= image[1], mask[1]
    image = np.expand_dims(sample_image, axis=0)
    activations = activation_model.predict(image)

    return activations


activations = get_model_activations(seg)

# Encoder layer visualizations

layer_names = ['conv1', 'conv_pw_1', 'conv_pw_4', 'conv_pw_7', 'conv_pw_10', 'conv_pw_13']
activ_list = [activations[2], activations[8], activations[28], activations[47], activations[65], activations[84]]
visualize_layer_activations(seg, layer_names, activ_list)

# Decoder layer visualizations

# Get conv2D layers
layer_names = []
for i in range(5):
  layer_names.append('decoder_stage{}a_conv'. format(i))
  layer_names.append('decoder_stage{}b_conv'. format(i))

# Get corresponding activations to those layers
activ_list = []
for l in layer_names:
  for idx, layer in enumerate(seg.model.layers):
    if layer.name == l:
        activ_list.append(activations[idx])
        break
  
visualize_layer_activations(seg, layer_names, activ_list)


#-------------------------------------------------
# ConvNet Filters
#-------------------------------------------------

# Transform filters into images
def deprocess_image(x):
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1
    x += 0.5
    x = np.clip(x, 0, 1)
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x

# Generate patterns for the given layer starting from an empty input image and
# applying Stochastic Gradient Ascent to maximize the response of a 
# particular filter
def generate_pattern(layer_name, filter_index, size=150):
    layer_output = seg.model.get_layer(layer_name).output
    loss = K.mean(layer_output[:, :, :, filter_index])
    grads = K.gradients(loss, seg.model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    iterate = K.function([seg.model.input], [loss, grads])
    input_img_data = np.random.random((1, size, size, 3)) * 20 + 128.
    step = 1.
    for i in range(80):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
        
    img = input_img_data[0]
    return deprocess_image(img)

# Visualize Convnet Filters 
def visualize_filter_grid(layer_name):
    size = 256
    margin = 5
    results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3))

    for i in range(8):
        for j in range(8):
            filter_img = generate_pattern(layer_name, i + (j * 8), size=size)
            print("generated filter {}". format(i * 8 + j))
            horizontal_start = i * size + i * margin
            horizontal_end = horizontal_start + size
            vertical_start = j * size + j * margin
            vertical_end = vertical_start + size
            results[horizontal_start: horizontal_end, vertical_start: vertical_end, :] = filter_img

    plt.figure(figsize=(15, 15))
    plt.title(layer_name)
    plt.imshow(results.astype(np.uint8))
    plt.savefig(VISUALIZATONPATH + layer_name + "_grid.jpg")

visualize_filter_grid('conv_pw_1')


#-------------------------------------------------
# Activation Maximization
#-------------------------------------------------

def visualize_activation_maximization(seg, n_examples=3):
    for j in range(n_examples):
        image, mask = next(seg.train_generator)
        sample_image, sample_mask = image[1], mask[1]
        plt.title("Sample Image")
        plt.imshow(sample_image)

        x = np.expand_dims(sample_image, axis=0)
        preds = seg.model.predict(x)
        output = seg.model.output[:,:,:, 0]

        last_conv_layer = seg.model.get_layer('final_conv')
        grads = K.gradients(output, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([seg.model.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([x])

        num_filters = 3
        for i in range(num_filters):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

        heatmap = np.mean(conv_layer_output_value, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        plt.title("Last Conv Layer {}".format(j))
        plt.imshow((heatmap*255).astype(np.uint8), cmap='plasma')
        plt.savefig(VISUALIZATONPATH + "activation_maximization{}.jpg".format(j))
        plt.show()

visualize_activation_maximization(seg)
