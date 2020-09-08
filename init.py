from keras_retinanet import models as retinanet_models
from tensorflow.keras import models as keras_models

vehicle_model_path = 'models/cars/cars_model_final.h5'
plate_model_path = 'models/plates/plates_model_final.h5'
char_model_path = 'models/chars/chars_model_big+drpt_fonts.h5'


def load_models():
    vehicle_model = retinanet_models.load_model(vehicle_model_path)
    plate_model = retinanet_models.load_model(plate_model_path)
    char_model = keras_models.load_model(char_model_path)

    return vehicle_model, plate_model, char_model