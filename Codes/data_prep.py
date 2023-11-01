from tensorflow.keras.utils import to_categorical

def data_prep(csv_data):
    img_height, img_width = 28, 28
    num_classes = 10
    channel = 1
    labels_one_hot = to_categorical(csv_data.label, num_classes)
    num_images = csv_data.shape[0]
    features_x = csv_data.values[:, 1:]
    img_feature = features_x.reshape(num_images, img_height, img_width, channel)
    features_x = img_feature / 255  # normalize
    return features_x, labels_one_hot
