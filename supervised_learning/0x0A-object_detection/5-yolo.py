#!/usr/bin/env python3
"""this module contains the Class Yolo"""
import tensorflow.keras as K
import numpy as np
import glob
import cv2


class Yolo:
    """
    Write a class Yolo that uses the Yolo v3 algorithm to perform object
     detection:
    In Class Constructor:h
        model_path is the path to where a Darknet Keras model is stored
        classes_path is the path to where the list of class names used for
         Darknet model, listed in order of index, can be found
        class_t is a float representing the box score threshold for the
         initial filtering step
        nms_t is a float representing the IOU threshold for non-max suppression
        anchors is a numpy.ndarray of shape (outputs, anchor_boxes, 2)
         containing all of the anchor boxes:
            outputs is the number of outputs (predictions) made by the Darknet
             model
            anchor_boxes is the number of anchor boxes used for each prediction
            2 => [anchor_box_width, anchor_box_height]
    Public instance attributes:
        model: the Darknet Keras model
        class_names: a list of the class names for the model
        class_t: the box score threshold for the initial filtering step
        nms_t: the IOU threshold for non-max suppression
        anchors: the anchor boxes
    """
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        self.model = K.models.load_model(model_path)
        self.class_names = []

        with open(classes_path, "r") as op:
            for x in op:
                self.class_names.append(x[0:-1])
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def sigmoid(self, x):
        """sigmoid function"""
        return 1 / (1 + np.exp(-x))

    def process_outputs(self, outputs, image_size):
        """
        outputs is a list of numpy.ndarrays containing the predictions from
         the Darknet model for a single image:
            Each output will have the shape (grid_height, grid_width,
             anchor_boxes, 4 + 1 + classes)
                grid_height & grid_width => the height and width of the grid
                 used for the output
                anchor_boxes => the number of anchor boxes used
                4 => (t_x, t_y, t_w, t_h)
                1 => box_confidence
                classes => class probabilities for all classes
        image_size is a numpy.ndarray containing the image’s original size
         [image_height, image_width]

        Returns a tuple of (boxes, box_confidences, box_class_probs):
            boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
             anchor_boxes, 4) containing the processed boundary boxes for each
             output, respectively:
            4 => (x1, y1, x2, y2)
            (x1, y1, x2, y2) should represent the boundary box relative to
             original image
            box_confidences: a list of numpy.ndarrays of shape (grid_height,
             grid_width, anchor_boxes, 1) containing the box confidences for
             each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape (grid_height,
             grid_width, anchor_boxes, classes) containing the box’s class
             probabilities for each output, respectively
        """
        image_height, image_width = image_size
        boxes = []
        box_confidences = []
        box_class_probs = []

        for n, box in enumerate(outputs):
            anchor = self.anchors[n]
            pw = anchor[:, 0]
            ph = anchor[:, 1]

            pw = pw.reshape((1, 1, len(pw)))
            ph = ph.reshape((1, 1, len(ph)))

            gh, gw, anchor_box, _ = box.shape

            # cx, cy - the grids values - no need for for loops
            cx = np.tile(np.arange(gw), gh).reshape(gw, gw, 1)
            cy = np.tile(np.arange(gh), gw).reshape(1, gw, gh).T

            t_x = box[:, :, :, 0]
            t_y = box[:, :, :, 1]
            t_w = box[:, :, :, 2]
            t_h = box[:, :, :, 3]

            bx = self.sigmoid(t_x) + cx
            by = self.sigmoid(t_y) + cy

            bw = pw * np.exp(t_w)
            bh = ph * np.exp(t_h)

            bx = bx/gw
            by = by/gh

            bw = bw / int(self.model.input.shape[1])
            bh = bh / int(self.model.input.shape[2])

            # x1, x2, y1, y2 - corners
            y1 = (by - bh/2) * image_height
            x1 = (bx - bw/2) * image_width
            x2 = (bw/2 + bx) * image_width
            y2 = (bh/2 + by) * image_height

            b_size = np.zeros((gh, gw, anchor_box, 4))
            b_size[:, :, :, 0] = x1
            b_size[:, :, :, 1] = y1
            b_size[:, :, :, 2] = x2
            b_size[:, :, :, 3] = y2
            boxes.append(b_size)

            # box confidences
            box_confidences.append(self.sigmoid(box[:, :, :, 4:5]))

            # box class
            box_class_probs.append(self.sigmoid(box[:, :, :, 5:]))

        return (boxes, box_confidences, box_class_probs)

    def filter_boxes(self, boxes, box_confidences, box_class_probs):
        """
            filter boxes - filters the preprossed boxes
            boxes: a list of numpy.ndarrays of shape (grid_height, grid_width,
             anchor_boxes, 4) containing the processed boundary boxes for each
              output, respectively
            box_confidences: a list of numpy.ndarrays of shape (grid_height,
             grid_width, anchor_boxes, 1) containing the processed box
              confidences for each output, respectively
            box_class_probs: a list of numpy.ndarrays of shape (grid_height,
             grid_width, anchor_boxes, classes) containing the processed box
              class probabilities for each output, respectively
            Returns a tuple of (filtered_boxes, box_classes, box_scores):
            filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
             the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,) containing the class
             number that each box in filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the box scores
             for each box in filtered_boxes, respectively
        """
        filtered_box = []
        prob_class = []
        box_c = []

        for i, bc in enumerate(box_confidences):
            gh, gw, ab, bi = bc.shape
            for h in range(gh):
                for w in range(gw):
                    for a in range(ab):
                        box = boxes[i][h, w, a]
                        b = bc[h, w, a, 0]
                        classes = box_class_probs[i][h, w, a]
                        bs = b * np.max(classes)
                        if bs >= self.class_t:
                            filtered_box.append(box)
                            prob_class.append(np.argmax(classes))
                            box_c.append(bs)
        box_scores = np.array(box_c)
        filtered_boxes = np.array(filtered_box)
        box_classes = np.array(prob_class)

        return (filtered_boxes, box_classes, box_scores)

    def nms(self, boxes, scores, threshold):
        """nms - applies non max suppression"""
        # filter indices to max scores
        order = scores.argsort()[::-1]

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        keep = []
        # iterate through indexes and do the suppression
        while order.size > 0:
            i = order[0]
            keep.append(i)

            tx1 = np.maximum(x1[i], x1[order[1:]])
            ty1 = np.maximum(y1[i], y1[order[1:]])
            tx2 = np.minimum(x2[i], x2[order[1:]])
            ty2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, tx2 - tx1 + 1)
            h = np.maximum(0.0, ty2 - ty1 + 1)

            inter = w * h
            IoU = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(IoU <= threshold)[0]
            order = order[inds + 1]
        return keep

    def non_max_suppression(self, filtered_boxes, box_classes, box_scores):
        """
            non max supression applies nms to the boxes

            filtered_boxes: a numpy.ndarray of shape (?, 4) containing all of
             the filtered bounding boxes:
            box_classes: a numpy.ndarray of shape (?,) containing the class
             number for the class that filtered_boxes predicts, respectively
            box_scores: a numpy.ndarray of shape (?) containing the box scores
             for each box in filtered_boxes, respectively
            Returns a tuple of (box_predictions, predicted_box_classes,
             predicted_box_scores):
            box_predictions: a numpy.ndarray of shape (?, 4) containing all of
             the predicted bounding boxes ordered by class and box score
            predicted_box_classes: a numpy.ndarray of shape (?,) containing
             the class number for box_predictions ordered by class and box
              score, respectively
            predicted_box_scores: a numpy.ndarray of shape (?) containing the
             box scores for box_predictions ordered by class and box score,
              respectively
        """
        keep_scores = []
        box, clas, score = [], [], []
        uniqux = np.unique(box_classes)

        for i in uniqux:
            indice = np.where(box_classes == i)

            boxes = filtered_boxes[indice]
            scores = box_scores[indice]
            classes = box_classes[indice]

            keep = self.nms(boxes, scores, self.nms_t,)

            box.append(boxes[keep])
            clas.append(classes[keep])
            score.append(scores[keep])

        box = np.concatenate(box)
        clas = np.concatenate(clas)
        score = np.concatenate(score)
        return (box, clas, score)

    @staticmethod
    def load_images(folder_path):
        """loads inmages using cv2"""
        images = []
        image_paths = [img for img in glob.glob(
            "{}/*.jpg".format(folder_path))]
        for img in image_paths:
            pho = cv2.imread(img)
            pho = np.array(pho)
            print("l pho", pho.shape)
            images.append(pho)
        return (images, image_paths)

    def preprocess_images(self, images):
        """preprocesses the imagaes"""
        ih = self.model.input.shape[1]
        iw = self.model.input.shape[2]
        ni = len(images)

        n = np.zeros((ni, ih, iw, 3))

        image_shapes = []
        for i in range(len(images)):
            image_shapes.append(images[i].shape[0:-1])

            pic = images[i] / 255
            rpic = cv2.resize(pic, (ih, iw), interpolation=cv2.INTER_CUBIC)
            n[i] = rpic

        image_shapes = np.array(image_shapes)
        pimages = n
        return (pimages, image_shapes)
