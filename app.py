import csv
import math
import os
import urllib.parse

import numpy as np
import tensorflow
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.models import load_model

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

UPLOAD_FOLDER = 'images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# define label meaning
label = ['apple pie',
         'baby back ribs',
         'baklava',
         'beef carpaccio',
         'beef tartare',
         'beet salad',
         'beignets',
         'bibimbap',
         'bread pudding',
         'breakfast burrito',
         'bruschetta',
         'caesar salad',
         'cannoli',
         'caprese salad',
         'carrot cake',
         'ceviche',
         'cheese plate',
         'cheesecake',
         'chicken curry',
         'chicken quesadilla',
         'chicken wings',
         'chocolate cake',
         'chocolate mousse',
         'churros',
         'clam chowder',
         'club sandwich',
         'crab cakes',
         'creme brulee',
         'croque madame',
         'cup cakes',
         'deviled eggs',
         'donuts',
         'dumplings',
         'edamame',
         'eggs benedict',
         'escargots',
         'falafel',
         'filet mignon',
         'fish and_chips',
         'foie gras',
         'french fries',
         'french onion soup',
         'french toast',
         'fried calamari',
         'fried rice',
         'frozen yogurt',
         'garlic bread',
         'gnocchi',
         'greek salad',
         'grilled cheese sandwich',
         'grilled salmon',
         'guacamole',
         'gyoza',
         'hamburger',
         'hot and sour soup',
         'hot dog',
         'huevos rancheros',
         'hummus',
         'ice cream',
         'lasagna',
         'lobster bisque',
         'lobster roll sandwich',
         'macaroni and cheese',
         'macarons',
         'miso soup',
         'mussels',
         'nachos',
         'omelette',
         'onion rings',
         'oysters',
         'pad thai',
         'paella',
         'pancakes',
         'panna cotta',
         'peking duck',
         'pho',
         'pizza',
         'pork chop',
         'poutine',
         'prime rib',
         'pulled pork sandwich',
         'ramen',
         'ravioli',
         'red velvet cake',
         'risotto',
         'samosa',
         'sashimi',
         'scallops',
         'seaweed salad',
         'shrimp and grits',
         'spaghetti bolognese',
         'spaghetti carbonara',
         'spring rolls',
         'steak',
         'strawberry shortcake',
         'sushi',
         'tacos',
         'octopus balls',
         'tiramisu',
         'tuna tartare',
         'waffles']

nu_link = 'https://www.nutritionix.com/food/'

# Loading the best saved model to make predictions.
tensorflow.keras.backend.clear_session()
model_best = load_model('best_model_101class.hdf5', compile=False)
print('model successfully loaded!')

nutrients = [
    {'name': 'protein', 'value': 0.0},
    {'name': 'calcium', 'value': 0.0},
    {'name': 'fat', 'value': 0.0},
    {'name': 'carbohydrates', 'value': 0.0},
    {'name': 'vitamins', 'value': 0.0}
]

with open('nutrition101.csv', 'r') as file:
    reader = csv.reader(file)
    nutrition_table = dict()
    for i, row in enumerate(reader):
        if i == 0:
            name = ''
            continue
        else:
            name = row[1].strip()
        nutrition_table[name] = [
            {'name': 'protein', 'value': float(row[2])},
            {'name': 'calcium', 'value': float(row[3])},
            {'name': 'fat', 'value': float(row[4])},
            {'name': 'carbohydrates', 'value': float(row[5])},
            {'name': 'vitamins', 'value': float(row[6])}
        ]


@app.route('/predict', methods=["POST"])
def predict():
    img_name = request.json['name']
    pa = dict()

    filename = f'{UPLOAD_FOLDER}/{img_name}'
    print('image filepath', filename)

    try:
        pred_img = image.load_img(filename, target_size=(200, 200))
    except FileNotFoundError:
        return dict()

    pred_img = image.img_to_array(pred_img)
    pred_img = np.expand_dims(pred_img, axis=0)
    pred_img = pred_img / 255.

    pred = model_best.predict(pred_img)

    if math.isnan(pred[0][0]) and math.isnan(pred[0][1]) and \
            math.isnan(pred[0][2]) and math.isnan(pred[0][3]):
        pred = np.array([0.05, 0.05, 0.05, 0.07, 0.09, 0.19, 0.55, 0.0, 0.0, 0.0, 0.0])

    top = pred.argsort()[0][-3:]
    label.sort()
    best_guest = label[top[2]]
    x = dict()
    x[best_guest] = float("{:.2f}".format(pred[0][top[2]] * 100))
    x[label[top[1]]] = float("{:.2f}".format(pred[0][top[1]] * 100))
    x[label[top[0]]] = float("{:.2f}".format(pred[0][top[0]] * 100))
    pa['result'] = x
    pa['nutrition'] = nutrition_table[best_guest]
    pa['food'] = f'{nu_link}{urllib.parse.quote(best_guest)}'
    pa['quantity'] = 100

    return jsonify(pa)


def lookup_env(env: str, default: str) -> str:
    result = os.getenv(env)
    if result is None:
        return default
    return result


if __name__ == "__main__":
    def run():
        host = lookup_env("HOST", "0.0.0.0")
        port = lookup_env("PORT", "3001")
        app.run(host=host, port=port)


    run()
