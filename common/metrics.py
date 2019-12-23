import torch
import torch.nn.functional as F

import common.utils as utils

__all__ = ['calc_mAP']


class MetricMAP():
    def __init__(self, device='cpu'):
        super(MetricMAP, self).__init__()

        self.device = device

        self.box_encoder = utils.BoxEncoder()

        self.num_classes = utils.labels_count()

        self.batch_count = 0
        self.all_average_precisions = {l: 0 for l in range(1, self.num_classes)}

    def calc_map(self, det_boxes, det_labels, det_scores, true_boxes, true_labels):
        assert len(det_boxes) == len(det_labels) == len(det_scores) == len(true_boxes) == len(true_labels)

        n_classes = self.num_classes

        true_images = list()
        for i in range(len(true_labels)):
            true_images.extend([i] * true_labels[i].size(0))
        true_images = torch.LongTensor(true_images).to(self.device)
        true_boxes = torch.cat(true_boxes, dim=0)
        true_labels = torch.cat(true_labels, dim=0)

        assert true_images.size(0) == true_boxes.size(0) == true_labels.size(0)

        det_images = list()
        for i in range(len(det_labels)):
            det_images.extend([i] * det_labels[i].size(0))
        det_images = torch.LongTensor(det_images).to(self.device)
        det_boxes = torch.cat(det_boxes, dim=0)
        det_labels = torch.cat(det_labels, dim=0)
        det_scores = torch.cat(det_scores, dim=0)

        assert det_images.size(0) == det_boxes.size(0) == det_labels.size(0) == det_scores.size(0)

        average_precisions = torch.zeros((n_classes - 1), dtype=torch.float32)
        for c in range(1, n_classes):
            true_class_images = true_images[true_labels == c]
            true_class_boxes = true_boxes[true_labels == c]

            true_class_boxes_detected = torch.zeros(
                (true_class_boxes.size(0)), dtype=torch.uint8).to(self.device)

            det_class_images = det_images[det_labels == c]
            det_class_boxes = det_boxes[det_labels == c]
            det_class_scores = det_scores[det_labels == c]
            n_class_detections = det_class_boxes.size(0)
            if n_class_detections == 0:
                continue

            det_class_scores, sort_ind = torch.sort(det_class_scores, dim=0, descending=True)
            det_class_images = det_class_images[sort_ind]
            det_class_boxes = det_class_boxes[sort_ind]

            true_positives = torch.zeros((n_class_detections), dtype=torch.float32).to(self.device)
            false_positives = torch.zeros((n_class_detections), dtype=torch.float32).to(self.device)
            for d in range(n_class_detections):
                this_detection_box = det_class_boxes[d].unsqueeze(0)
                this_image = det_class_images[d]

                object_boxes = true_class_boxes[true_class_images == this_image]
                if object_boxes.size(0) == 0:
                    false_positives[d] = 1
                    continue

                overlaps = utils.calc_iou_tensor(this_detection_box, object_boxes)
                max_overlap, ind = torch.max(overlaps.squeeze(0), dim=0)

                original_ind = torch.LongTensor(
                    range(true_class_boxes.size(0)))[true_class_images == this_image][ind]

                if max_overlap.item() > 0.5:
                    if true_class_boxes_detected[original_ind] == 0:
                        true_positives[d] = 1
                        true_class_boxes_detected[original_ind] = 1
                    else:
                        false_positives[d] = 1
                else:
                    false_positives[d] = 1

            cumul_true_positives = torch.cumsum(true_positives, dim=0)
            cumul_false_positives = torch.cumsum(false_positives, dim=0)
            cumul_precision = cumul_true_positives / (
                    cumul_true_positives + cumul_false_positives + 1e-10)
            cumul_recall = cumul_true_positives / torch.tensor(
                true_class_images.shape).prod().item()

            recall_thresholds = torch.arange(start=0, end=1.1, step=.1).tolist()
            precisions = torch.zeros((len(recall_thresholds)), dtype=torch.float32).to(self.device)
            for i, t in enumerate(recall_thresholds):
                recalls_above_t = cumul_recall >= t
                if recalls_above_t.any():
                    precisions[i] = cumul_precision[recalls_above_t].max()
                else:
                    precisions[i] = 0.
            average_precisions[c - 1] = precisions.mean()

        average_precisions = {c + 1: v for c, v in enumerate(average_precisions.tolist())}

        return average_precisions

    def add(self, det_boxes, det_scores, true_boxes, true_labels):
        out = self.box_encoder.decode_batch(det_boxes, det_scores)
        det_boxes, det_labels, det_scores = zip(*out)
        det_labels = list(det_labels)
        for i in range(len(det_labels)):
            det_boxes[i] = det_boxes[i].to(self.device)
            det_labels[i] = det_labels[i].type(torch.int64).to(self.device)

        true_scores = F.one_hot(true_labels, num_classes=self.num_classes)
        true_scores = (true_scores * 1e9).type(torch.float32)
        true_scores = true_scores.permute(0, 2, 1)
        out = self.box_encoder.decode_batch(true_boxes, true_scores)
        true_boxes, true_labels, _ = zip(*out)

        average_precisions = self.calc_map(det_boxes, det_labels, det_scores, 
                                           true_boxes, true_labels)

        self.batch_count += 1
        for l, v in average_precisions.items():
            self.all_average_precisions[l] += v

    def get(self):
        average_precisions = self.all_average_precisions.copy()

        vs = self.all_average_precisions.values()
        map_value = sum(vs) / self.batch_count / len(vs)

        return average_precisions, map_value