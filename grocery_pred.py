import numpy as np
import tensorflow as tf
from speech import sayAudio, requestUser
from threading import Thread
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from object_detection.utils import config_util

groceryItems = []
previous = ""
label = ""
item = "Welcome to vAssist, Let's get started!"
speak = True
isOccupied = True  # currently at an aisle or not

CONFIG_PATH = '/Users/ShrinandanNarayanan/Projects/vAssist/ComputerVision/GroceryDetectionNN/my_ssd_mobnet/pipeline.config'
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore('GroceryDetectionNN/my_ssd_mobnet/ckpt-6').expect_partial()
category_index = label_map_util.create_category_index_from_labelmap(
    'GroceryDetectionNN/data/train/Grocery-Items_label_map.pbtxt')

#  SET THE SHOPPING LIST
def setItems(items):
    global groceryItems
    groceryItems = items


def sayItem():
    global item
    sayAudio(item)


def itemThread(item):
    item_thread = Thread(target=item, daemon=True)
    item_thread.start()


#  CRITERIA TO LAUNCH BARCODE SCANNER
def isProduct(item):
    # sample products
    prods = ['bread', 'cinnamon', 'ketchup', 'milk', 'juice']
    if item in prods:
        return True
    else:
        return False


#  GET PREDICTIONS ON INPUT FRAME
@tf.function
def detect_item(image):
    global speak
    # image = jetson.utils.cudaFromNumpy(image) -> leverages CUDA
    images, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(images, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)

    return detections


def get_preds(frame):
    global previous
    global label
    global item
    global speak
    global groceryItems
    global isOccupied

    num = 1  # 0 -> grocery item recog, 1 -> barcode scanner
    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    image_np_with_detections = image_np.copy()

    detections = detect_item(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                  for key, value in detections.items()}
    boxes = detections['detection_boxes']
    classes = detections['detection_classes'].astype(np.int64)
    scores = detections['detection_scores']

    detections['num_detections'] = num_detections

    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1

    #  LOOP THROUGH BOXES
    for i, b in enumerate(boxes[0]):
        if (scores[i] < 0.8):
            label = "No item found" + ", " + str(scores[0])
            item = "No item found"
        else:
            label = str(category_index[classes[0] + 1]) + ", " + str(
                scores[0] * 100) + "%"
            item = str(category_index[detections['detection_classes'][0] + 1]['name'])
            if (item != previous):
                previous = item
                # sayAudio(item)
                # itemThread(sayItem)
                print(item)
                if item.lower() == str(groceryItems[0]).lower():
                    viz_utils.visualize_boxes_and_labels_on_image_array(
                        image_np_with_detections,
                        np.squeeze(boxes),
                        np.squeeze(classes + label_id_offset),
                        np.squeeze(scores),
                        category_index,
                        use_normalized_coordinates=True,
                        max_boxes_to_draw=5,
                        min_score_thresh=.8,
                        agnostic_mode=False,
                        line_thickness=8)
                    groceryItems.remove(item)
                    if isProduct(item.lower()):
                        itemThread(sayItem)
                        req = requestUser("This item is on your shopping list, would you like to scan the barcode?")
                        if req == "yes":
                            item = "Launching barcode scanner"
                            itemThread(sayItem)
                            num = 2
                    else:
                        request = requestUser("Would you like to continue scanning grocery items?")
                        if request == 'yes':
                            item = "Ok, please continue scanning"
                            itemThread(item)
                            num = 1
                        else:
                            item = "Ok, please continue shopping"
                            itemThread(sayItem)
                            isOccupied = False
                            num = 0

    return image_np_with_detections, num, isOccupied

