import cv2
import numpy as np
import tensorflow as tf
from DepthEstimation.estimation import set_boundary

#  LOAD MIDAS INTERPRETER
interpreter = tf.lite.Interpreter(model_path="DepthEstimation/model_opt.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

prediction = []
depth_max = 0

#  PREPROCESS THE FRAME
def process_image(frame):
    frame = cv2.GaussianBlur(frame, (5, 5),0)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
    frame_resized = tf.image.resize(frame, [256, 256], method='bicubic', preserve_aspect_ratio=False)
    # img_resized = tf.transpose(img_resized, [2, 0, 1])
    frame_input = frame_resized.numpy()
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    frame_input = (frame_input - mean) / std
    reshape_frame = frame_input.reshape(1, 256, 256, 3)
    tensor = tf.convert_to_tensor(reshape_frame, dtype=tf.float32)

    return tensor

# PERFORM INFERENCE ON FRAME
def inference(tensor):
    global interpreter
    interpreter.set_tensor(input_details[0]['index'], tensor)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    output = output.reshape(256, 256)
    return output

# RETRIEVE INVERSE DEPTH MAP
def predict(frame):
    global depth_max
    global prediction
    output = inference(process_image(frame))
    prediction = cv2.resize(output, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
    depth_min = prediction.min()
    depth_max = prediction.max()
    frame_out = (255 * (prediction - depth_min) / (depth_max - depth_min)).astype("uint8")
    # print('prediction: {}'.format(prediction))
    # print('depth_min: {}'.format(depth_min))
    # print('depth_max: {}'.format(depth_max))
    return frame_out

#  GET COORDS OF INTEREST (CLOSE REGIONS)
def get_coords(frame):
    global depth_max
    thres = 140
    locs_max = np.where(frame >= thres)
    locs_min = np.where(frame < thres)
    coords_max = list(zip(locs_max[0], locs_max[1]))
    coords_min = list(zip(locs_min[0], locs_min[1]))
    return  coords_min, coords_max

#  MASK OUT THE CLOSE REGIONS
def mask(depth, frame, width, height):
    coords_min, coords_max = get_coords(depth)
    masked1 = frame.copy()
    masked2 = frame.copy()
    for coord in coords_max:
        x = coord[0]
        y = coord[1]
        masked1[x, y] = (204, 91, 210)

    for coord in coords_min:
        x = coord[0]
        y = coord[1]
        masked2[x, y] = (0, 0, 0)

    frame = set_boundary(frame, width, height)

    return masked2



