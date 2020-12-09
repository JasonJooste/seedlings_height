"""
A file containing modifications to some functions of the torchvision packages
"""
from torchvision.models.detection.roi_heads import fastrcnn_loss, RoIHeads
import torch
from torch import nn, Tensor
from torch.jit.annotations import Optional, List, Dict, Tuple


# Taken and modified from torchvision.models.detection.roi_heads
# def __init__(self,
#              box_roi_pool,
#              box_head,
#              box_predictor,
#              # Faster R-CNN training
#              fg_iou_thresh, bg_iou_thresh,
#              batch_size_per_image, positive_fraction,
#              bbox_reg_weights,
#              # Faster R-CNN inference
#              score_thresh,
#              nms_thresh,
#              detections_per_img,
#              # Mask
#              mask_roi_pool=None,
#              mask_head=None,
#              mask_predictor=None,
#              keypoint_roi_pool=None,
#              keypoint_head=None,
#              keypoint_predictor=None,
#              ):
#     super(RoIHeads, self).__init__()
#
#     self.box_similarity = box_ops.box_iou
#     # assign ground-truth boxes for each proposal
#     self.proposal_matcher = det_utils.Matcer(
#         fg_iou_thresh,
#         bg_iou_thresh,
#         allow_low_quality_matches=False)
#
#     self.fg_bg_sampler = det_utils.BalancedPositiveNegativeSampler(
#         batch_size_per_image,
#         positive_fraction)
#
#     if bbox_reg_weights is None:
#         bbox_reg_weights = (10., 10., 5., 5.)
#     self.box_coder = det_utils.BoxCoder(bbox_reg_weights)
#
#     self.box_roi_pool = box_roi_pool
#     self.box_head = box_head
#     self.box_predictor = box_predictor
#
#     self.score_thresh = score_thresh
#     self.nms_thresh = nms_thresh
#     self.detections_per_img = detections_per_img
#
#     self.mask_roi_pool = mask_roi_pool
#     self.mask_head = mask_head
#     self.mask_predictor = mask_predictor
#
#     self.keypoint_roi_pool = keypoint_roi_pool
#     self.keypoint_head = keypoint_head
#     self.keypoint_predictor = keypoint_predictor
class RoIHeadsWrapper(RoIHeads):
    def __init__(self, ip):
        """
        Build the wrapper instance from an existing RoIHeads instance
        :param ip:
        """
        assert isinstance(ip, RoIHeads), "The input should be a RoIHeads instance"
        super(RoIHeadsWrapper, self).__init__(ip.box_roi_pool,
                                              ip.box_head,
                                              ip.box_predictor,
                                              ip.proposal_matcher.high_threshold,
                                              ip.proposal_matcher.low_threshold,
                                              ip.fg_bg_sampler.batch_size_per_image,
                                              ip.fg_bg_sampler.positive_fraction,
                                              ip.box_coder.weights,
                                              ip.score_thresh,
                                              ip.nms_thresh,
                                              ip.detections_per_img)

    def forward(self,
                features,      # type: Dict[str, Tensor]
                proposals,     # type: List[Tensor]
                image_shapes,  # type: List[Tuple[int, int]]
                targets=None   # type: Optional[List[Dict[str, Tensor]]]
                ):
        # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
        """
        Arguments:
            features (List[Tensor])
            proposals (List[Tensor[N, 4]])
            image_shapes (List[Tuple[H, W]])
            targets (List[Dict])
        """
        if targets is not None:
            for t in targets:
                # TODO: https://github.com/pytorch/pytorch/issues/26731
                floating_point_types = (torch.float, torch.double, torch.half)
                assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
                assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
                if self.has_keypoint():
                    assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'

        if self.training:
            # Get proposals and match them to targets
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None
        # Extract the feature maps for each proposal using roi pooling
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        # This has two fully connected layers
        box_features = self.box_head(box_features)
        # We now have the two heads - one for classification and one for the boxes of each class
        class_logits, box_regression = self.box_predictor(box_features)

        result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
        losses = {}
        if self.training:
            assert labels is not None and regression_targets is not None
            # Calculate the loss
            loss_classifier, loss_box_reg = fastrcnn_loss(
                class_logits, box_regression, labels, regression_targets)
            losses = {
                "loss_classifier": loss_classifier,
                "loss_box_reg": loss_box_reg
            }
        # else:
        boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        num_images = len(boxes)
        for i in range(num_images):
            result.append(
                {
                    "boxes": boxes[i],
                    "labels": labels[i],
                    "scores": scores[i],
                }
            )
        # We are not dealing with masks or keypoints
        assert not self.has_mask(), "This functionality is not supported"
        assert self.keypoint_roi_pool is None, "This functionality is not supported"
        return result, losses
