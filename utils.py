import tensorflow as tf

thirty_five_class_names = ['Hainanese Chicken Rice', 'apple', 'bak chor mee', 'bak kut teh',
       'ban mian soup', 'banana', 'char kway teow', 'chendol',
       'chicken curry noodle', 'claypot rice', 'curry puff',
       'fish head curry', 'fried carrot cake', 'grapes', 'ice kacang',
       'kiwi', 'laksa', 'mango', 'mee hoon kueh', 'mee rebus', 'mee siam',
       'nasi briyani', 'nasi lemak', 'orange', 'oyster omelette', 'pear',
       'pineapple', 'pomegranate', 'popiah', 'prawn noodles soup',
       'roti john', 'roti prata', 'satay', 'wanton mee', 'watermelon']

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
            'Satay':'https://github.com/DSstore/AIP/raw/main/Satay%20(4).jpeg',
            'Bak chor mee': 'https://github.com/DSstore/AIP/raw/main/BAK%20CHOR%20MEE%20DRY.jpg',
            'Ban mian soup': 'https://github.com/DSstore/AIP/raw/main/BAN%20MIAN.jpg',
            'Chicken curry noodle': 'https://github.com/DSstore/AIP/raw/main/CURRY%20CHICKEN.jpg',
            'Claypot rice': 'https://github.com/DSstore/AIP/raw/main/CLAYPOT%20RICE.jpg',
            'Fried carrot cake': 'https://github.com/DSstore/AIP/raw/main/FRIED%20CARROT%20CAKE.jpg',
            'Mee hoon kueh': 'https://github.com/DSstore/AIP/raw/main/MEE%20HOON%20KWAY.jpg',
            'Mee rebus': 'https://github.com/DSstore/AIP/raw/main/MEE%20REBUS.jpg',
            'Mee siam': 'https://github.com/DSstore/AIP/raw/main/MEE%20SIAM.jpg',
            'Nasi briyani': 'https://github.com/DSstore/AIP/raw/main/NASI%20BRIYANI.jpg',
            'Popiah': 'https://github.com/DSstore/AIP/raw/main/POPIAH.jpg',
            'Prawn noodles soup': 'https://github.com/DSstore/AIP/raw/main/PRAWN%20NOODLES%20SOUP.jpg',
            'Roti john': 'https://github.com/DSstore/AIP/raw/main/ROTI%20JOHN.jpg',
            'Wanton mee': 'https://github.com/DSstore/AIP/raw/main/WONTON%20NOODLES.jpg',
            'Ice kacang': 'https://github.com/DSstore/AIP/raw/main/ICE%20KACHANG.jpg',
            'Fish head curry': 'https://github.com/DSstore/AIP/raw/main/FISH%20HEAD%20CURRY.jpg'}

def get_2_classes():
    return two_class_names

def get_10_classes():
    return ten_class_names

def get_20_classes():
    return twenty_class_names

def get_35_classes():
    return thirty_five_class_names

def get_calories():
    return calories

def load_and_prep(image, shape=224, scale=False):
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, size=([shape, shape]))
    if scale:
        image = image/255.
    return image

