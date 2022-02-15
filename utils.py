import datetime
import tensorflow as tf

twenty_class_names = ['Hainanese Chicken Rice',
                'apple',
                'bak kut teh',
                'banana',
                'char kway teow',
                'chendol',
                'curry puff',
                'grapes',
                'kiwi',
                'laksa',
                'mango',
                'nasi lemak',
                'orange',
                'oyster omelette',
                'pear',
                'pineapple',
                'pomegranate',
                'roti prata',
                'satay',
                'watermelon']

ten_class_names = ['Hainanese Chicken Rice',
                'bak kut teh',
                'char kway teow',
                'chendol',
                'curry puff',
                'laksa',
                'nasi lemak',
                'oyster omelette',
                'roti prata',
                'satay']

two_class_names = ['laksa',
                   'satay']



calories = {'Hainanese chicken rice': 'https://github.com/DSstore/AIP/raw/main/Hainanesechickenrice%20(4).jpeg',
            'Bak kut teh': 'https://github.com/DSstore/AIP/raw/main/Bakkutteh%20(5).jpeg',
            'Char kway teow':'https://github.com/DSstore/AIP/raw/main/Charkwayteow%20(3).jpeg',
            'Chendol':'https://github.com/DSstore/AIP/raw/main/Cendol%20(4).jpeg',
            'Curry puff':'https://github.com/DSstore/AIP/raw/main/Currypuff%20(4).jpeg',
            'Laksa':'https://github.com/DSstore/AIP/raw/main/Laksa%20(4).jpeg',
            'Nasi lemak':'https://github.com/DSstore/AIP/raw/main/Nasilemak%20(4).jpeg',
            'Oyster omelette':'https://github.com/DSstore/AIP/raw/main/Oysteromelette.jpeg',
            'Roti prata':'https://github.com/DSstore/AIP/raw/main/Rotiprata%20(4).jpeg',
            'Satay':'https://github.com/DSstore/AIP/raw/main/Satay%20(4).jpeg'}

def get_2_classes():
    return two_class_names

def get_10_classes():
    return ten_class_names

def get_20_classes():
    return twenty_class_names

def get_calories():
    return calories

def load_and_prep(image, shape=224, scale=False):
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, size=([shape, shape]))
    if scale:
        image = image/255.
    return image

