# ======================================================================================================================
# name        : tensorboard.py
# type        : tensorboard summary functions
# purpose     : define what information should be stored in summary and displayed in tensorboard
# author      : Olga Dergachyova
# last update : 10/2020
# ======================================================================================================================

import tensorflow as tf
import numpy as np


# saves information about weight and bias gradients
def gradients_summary(cost, variables):
    kernels = [v for v in variables if 'kernel' in v.name]
    biases = [v for v in variables if 'bias' in v.name]

    summaries = []
    with tf.name_scope("weight_gradients"):
        for k in kernels:
            sep = '_' if 'deconv' in k.name else '-'
            name = k.name.split('/')[0] + sep + k.name.split('/')[1].split(':')[0]
            grad = tf.gradients(cost, k)[0]
            mean = tf.reduce_mean(tf.abs(grad))
            summaries.append(tf.summary.scalar(name + '-mean', mean))
            summaries.append(tf.summary.histogram(name, grad))

    with tf.name_scope("bias_gradients"):
        for b in biases:
            sep = '_' if 'deconv' in b.name else '-'
            name = b.name.split('/')[0] + sep + b.name.split('/')[1].split(':')[0]
            grad = tf.gradients(cost, b)[0]
            mean = tf.reduce_mean(tf.abs(grad))
            summaries.append(tf.summary.scalar(name + '-mean', mean))
            summaries.append(tf.summary.histogram(name, grad))

    return tf.summary.merge(summaries)


# saves information about weights and biases
def variables_summary(variables):
    kernels = [v for v in variables if 'kernel' in v.name]
    biases = [v for v in variables if 'bias' in v.name]

    summaries = []
    # -------- Hists -------- #
    with tf.name_scope("weights"):
        for k in kernels:
            # if ('flat' in k.name) or ('output' in k.name) or ('conv2' in k.name):
            sep = '_' if 'deconv' in k.name else '-'
            name = k.name.split('/')[0] + sep + k.name.split('/')[1].split(':')[0]
            summaries.append(tf.summary.histogram(name, k))

    with tf.name_scope("biases"):
        for b in biases:
            # if ('flat' in k.name) or ('output' in k.name) or ('conv2' in k.name):
            sep = '_' if 'deconv' in b.name else '-'
            name = b.name.split('/')[0] + sep + b.name.split('/')[1].split(':')[0]
            summaries.append(tf.summary.histogram(name, b))

    return tf.summary.merge(summaries)


# saves visual information about weights and biases as images
def variables_image_summary(variables):
    kernels = [v for v in variables if 'kernel' in v.name]

    summaries = []

    with tf.name_scope("weights_channels_mean"):
        for k in kernels:
            if ('flat' not in k.name) and ('output' not in k.name):
                sep = '_' if 'deconv' in k.name else '-'
                name = k.name.split('/')[0] + sep + k.name.split('/')[1].split(':')[0]
                summaries.append(tf.summary.image(name, var_tensor_to_image(k, mean=True)))

    with tf.name_scope("weights_channels_sum"):
        for k in kernels:
            if ('flat' not in k.name) and ('output' not in k.name):
                sep = '_' if 'deconv' in k.name else '-'
                name = k.name.split('/')[0] + sep + k.name.split('/')[1].split(':')[0]
                summaries.append(tf.summary.image(name, var_tensor_to_image(k)))

    return tf.summary.merge(summaries)


# saves information about activation functions
def activations_summary(layers, network):
    if network == "unet":
        return activations_summary_unet(layers)
    else:
        return activations_summary_offresnet(layers)


# saves information about filters
def features_summary(convs, deconvs, network):
    if network == "unet":
        return features_summary_unet(convs, deconvs)
    else:
        return features_summary_offresnet(convs)


# saves information about activation functions in offresnet
def activations_summary_offresnet(layers):
    summaries = []
    with tf.name_scope("activations"):
        for layer in layers.keys():
            if ('input' in layer) or ('flat' in layer) or ('output' in layer):
                summaries.append(tf.summary.histogram(layer, layers[layer][0]))
            else:
                for i, l in enumerate(layers[layer]):
                    summaries.append(tf.summary.histogram(layer + '-conv' + str(i+1), l))
    return tf.summary.merge(summaries)


# saves information about filters in offresnet
def features_summary_offresnet(convs):
    summaries = []
    with tf.name_scope("features"):
        for layer in convs.keys():
            if ('input' in layer) or ('flat' in layer) or ('output' in layer):
                summaries.append(tf.summary.histogram(layer, convs[layer][0]))
            else:
                for i, l in enumerate(convs[layer]):
                    summaries.append(tf.summary.histogram(layer + '-conv' + str(i+1), l))
    return tf.summary.merge(summaries)


# saves information about activation functions in unet
def activations_summary_unet(layers):
    summaries = []
    with tf.name_scope("activations"):
        for layer in layers.keys():
            if 'down' in layer:
                summaries.append(tf.summary.histogram(layer + '-conv1', layers[layer][0]))
                summaries.append(tf.summary.histogram(layer + '-conv2', layers[layer][1]))
            elif 'up' in layer:
                summaries.append(tf.summary.histogram(layer + '_deconv', layers[layer][0]))
                summaries.append(tf.summary.histogram(layer + '-conv1', layers[layer][1]))
                summaries.append(tf.summary.histogram(layer + '-conv2', layers[layer][2]))
            elif 'flat' in layer or 'output' in layer:
                layer_ = layers[layer][0]
                shape = layer_.get_shape().as_list()
                if shape[-1] > 1:
                    for i in range(shape[-1]):
                        summaries.append(tf.summary.histogram(layer + '-ch' + str(i+1), layer_[i]))
                else:
                    summaries.append(tf.summary.histogram(layer, layers[layer][0]))
    return tf.summary.merge(summaries)


# saves information about filters in unet
def features_summary_unet(convs, deconvs):
    summaries = []
    with tf.name_scope("features"):
        for layer in convs.keys():
            if 'down' in layer or 'up' in layer:
                summaries.append(tf.summary.histogram(layer + '-conv1', convs[layer][0]))
                summaries.append(tf.summary.histogram(layer + '-conv2', convs[layer][1]))
            elif 'flat' in layer or 'output' in layer:
                layer_ = convs[layer][0]
                shape = layer_.get_shape().as_list()
                if shape[-1] > 1:
                    for i in range(shape[-1]):
                        summaries.append(tf.summary.histogram(layer + '-ch' + str(i+1), layer_[i]))
                else:
                    summaries.append(tf.summary.histogram(layer, convs[layer][0]))
        for layer in deconvs.keys():
                summaries.append(tf.summary.histogram(layer + '_deconv', deconvs[layer]))
    return tf.summary.merge(summaries)


# saves visual information about activation functions as images
def activations_image_summary(activs):
    summaries = []
    with tf.name_scope("activations_images"):
        for k in activs.keys():
            if isinstance(activs[k], tuple):
                for i, c in enumerate(activs[k]):
                    (features, tensor) = tensor_to_image(c)
                    sep = '_' if 'deconv' in c.name else '-'
                    name = k + sep + c.name.split('/')[1].split(':')[0]
                    summaries.append(tf.summary.image(name, tensor, max_outputs=features))
            else:
                (features, tensor) = tensor_to_image(activs[k])
                name = k + '-' + activs[k].name.split('/')[1].split(':')[0]
                summaries.append(tf.summary.image(name, tensor, max_outputs=features))
    return tf.summary.merge(summaries)


# saves visual information about filters as images
def features_image_summary(convs, deconvs):
    summaries = []
    with tf.name_scope("features_images"):
        for k in convs.keys():
            if isinstance(convs[k], tuple):
                for i, c in enumerate(convs[k]):
                    (features, tensor) = tensor_to_image(c)
                    summaries.append(tf.summary.image(k + '-conv' + str(i+1), tensor, max_outputs=features))
            else:
                (features, tensor) = tensor_to_image(convs[k])
                summaries.append(tf.summary.image(k, tensor, max_outputs=features))

        for k in deconvs.keys():
            (features, tensor) = tensor_to_image(deconvs[k])
            summaries.append(tf.summary.image(k + '_deconv', tensor, max_outputs=features))
    return tf.summary.merge(summaries)


# transforms multi-dimensional tensor to images
def var_tensor_to_images(tensor):
    if len(tensor.get_shape().as_list()) == 5:
        tensor = tf.slice(tensor, (0, 0, 0, 0, 0), (-1, -1, 1, -1, -1))
        [h, w, _, c, f] = tensor.get_shape().as_list()
        tensor = tf.reshape(tensor, (h, w, c, f))

    [h, w, c, f] = tensor.get_shape().as_list()
    max = tf.reduce_max(tensor)
    bh = h + 2
    bw = w + 2

    # for each feature, make image of its channels
    features = []
    for i in range(0,f):
        t = tf.slice(tensor, (0, 0, 0, i), (-1, -1, -1, 1))
        t = tf.reshape(t, (h, w, c))
        t = tf.image.resize_image_with_crop_or_pad(t, bh, bw)
        if c >= 8:
            nx = int(8)
        elif (c < 8) and (c >=4):
            nx = int(4)
        else:
            nx = c
        ny = int(np.ceil(c/nx))
        t = tf.reshape(t, (bh,bw,ny,nx))
        t = tf.transpose(t, (2, 0, 3, 1))
        t = tf.reshape(t, (bh*ny, bw*nx))
        features.append(t)

    # add white space after each feature
    for i in range(0,f-1):
        feature = features[i]
        [h, w] = feature.get_shape().as_list()
        tmp = tf.fill([4,w], max)
        features[i] = tf.concat([feature, tmp], 0)

    # put all features on the same image
    t = tf.concat(features, 0)
    [h, w] = t.get_shape().as_list()
    t = tf.reshape(t, (1, h, w, 1))
    return t


# transforms tensor to image
def var_tensor_to_image(tensor, mean=False):
    if len(tensor.get_shape().as_list()) == 5:
        tensor = tf.slice(tensor, (0, 0, 0, 0, 0), (-1, -1, 1, -1, -1))
        [h, w, _, c, f] = tensor.get_shape().as_list()
        tensor = tf.reshape(tensor, (h, w, c, f))

    if mean is True:
        t = tf.reduce_mean(tensor, 2)
    else:
        t = tf.reduce_sum(tensor, 2)
    [h, w, f] = t.get_shape().as_list()
    t = tf.reshape(t, (h,w,f))
    bh = h + 2
    bw = w + 2

    if f == 1:
        return tf.reshape(t, (1, h, w, 1))
    else:
        nx = int(8)
        ny = int(np.ceil(f/nx))
        t = tf.image.resize_image_with_crop_or_pad(t, bh, bw)
        t = tf.reshape(t, (bh,bw,ny,nx))
        t = tf.transpose(t, (2, 0, 3, 1))
        t = tf.reshape(t, (1, bh*ny, bw*nx, 1))
        return t


# transforms tensor to image
def tensor_to_image(tensor):
    if len(tensor.get_shape().as_list()) == 5:
        t = tf.slice(tensor, (0, 0, 0, 0, 0), (1, -1, -1, 1, -1))
        [_,h,w,_,f] = t.get_shape().as_list()
        t = tf.reshape(t, (h,w,f))
    else:
        t = tf.slice(tensor, (0, 0, 0, 0), (1, -1, -1, -1))
        [_,h,w,f] = t.get_shape().as_list()
        t = tf.reshape(t, (h,w,f))
    if f >= 4:
        nx = int(4)
        ny = int(np.ceil(f/nx))
        t = tf.reshape(t, (h,w,ny,nx))
        t = tf.transpose(t, (2, 0, 3, 1))
        t = tf.reshape(t, (1, h*ny, w*nx, 1))
        return 1, t
    else:
        t = tf.transpose(t, (2, 0, 1))
        t = tf.reshape(t, (f, h, w, 1))
        return f, t