"""
A file containing modifications to some functions of the torchvision packages
"""
from collections import OrderedDict

import warnings
from typing import Union

import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.roi_heads import fastrcnn_loss, RoIHeads
import torch
from torch import nn, Tensor
from torch.jit.annotations import Optional, List, Dict, Tuple
import torch.nn.functional as F
from torchvision.ops import roi_align
from torchvision.ops.poolers import _onnx_merge_levels, initLevelMapper, LevelMapper, MultiScaleRoIAlign


class RoIHeadsFinalLayer(RoIHeads):
    def forward(self,
                features,      # type: Dict[str, Tensor]
                heights,       # type: [Tensor]
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
        attrs = vars(self.box_roi_pool)
        box_features = self.box_roi_pool(features, proposals, image_shapes)
        ### Now we extract the features from the heights
        # First the heights need to be in the correct format.
        heights = [height_im.unsqueeze(0) for height_im in heights]
        heights = torch.cat(heights, 0)
        heights = {"heights": heights}
        box_heights = self.box_roi_pool(heights, proposals, image_shapes)
        # Now we concatenate the features together
        new_features = torch.cat((box_heights, box_features), 1)
        # This has two fully connected layers
        box_features = self.box_head(new_features)
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


class FasterRCNNEndHeights(FasterRCNN):
    def forward(self, images, heights, targets=None):
        # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        if self.training:
            assert targets is not None
            for target in targets:
                boxes = target["boxes"]
                if isinstance(boxes, torch.Tensor):
                    if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                        raise ValueError("Expected target boxes to be a tensor"
                                         "of shape [N, 4], got {:}.".format(
                                             boxes.shape))
                else:
                    raise ValueError("Expected target boxes to be of type "
                                     "Tensor, got {:}.".format(type(boxes)))

        original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
        for img in images:
            val = img.shape[-2:]
            assert len(val) == 2
            original_image_sizes.append((val[0], val[1]))

        images, targets = self.transform(images, targets)

        # Check for degenerate boxes
        # TODO: Move this to a function
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    raise ValueError("All bounding boxes should have positive height and width."
                                     " Found invalid box {} for target at index {}."
                                     .format(degen_bb, target_idx))

        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        # This is the only difference to the forward function - we feed the heights parameter in here
        detections, detector_losses = self.roi_heads(features, heights, proposals, images.image_sizes, targets)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("RCNN always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return (losses, detections)
        else:
            return self.eager_outputs(losses, detections)



# class MultiScaleRoIAlignTemp(nn.Module):
#     """
#     Multi-scale RoIAlign pooling, which is useful for detection with or without FPN.
#
#     It infers the scale of the pooling via the heuristics present in the FPN paper.
#
#     Arguments:
#         featmap_names (List[str]): the names of the feature maps that will be used
#             for the pooling.
#         output_size (List[Tuple[int, int]] or List[int]): output size for the pooled region
#         sampling_ratio (int): sampling ratio for ROIAlign
#
#     Examples::
#
#         >>> m = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat3'], 3, 2)
#         >>> i = OrderedDict()
#         >>> i['feat1'] = torch.rand(1, 5, 64, 64)
#         >>> i['feat2'] = torch.rand(1, 5, 32, 32)  # this feature won't be used in the pooling
#         >>> i['feat3'] = torch.rand(1, 5, 16, 16)
#         >>> # create some random bounding boxes
#         >>> boxes = torch.rand(6, 4) * 256; boxes[:, 2:] += boxes[:, :2]
#         >>> # original image size, before computing the feature maps
#         >>> image_sizes = [(512, 512)]
#         >>> output = m(i, [boxes], image_sizes)
#         >>> print(output.shape)
#         >>> torch.Size([6, 5, 3, 3])
#
#     """
#
#     __annotations__ = {
#         'scales': Optional[List[float]],
#         'map_levels': Optional[LevelMapper]
#     }
#
#     def __init__(
#         self,
#         featmap_names: List[str],
#         output_size: Union[int, Tuple[int], List[int]],
#         sampling_ratio: int,
#     ):
#         super(MultiScaleRoIAlign, self).__init__()
#         if isinstance(output_size, int):
#             output_size = (output_size, output_size)
#         self.featmap_names = featmap_names
#         self.sampling_ratio = sampling_ratio
#         self.output_size = tuple(output_size)
#         self.scales = None
#         self.map_levels = None
#
#     def convert_to_roi_format(self, boxes: List[Tensor]) -> Tensor:
#         concat_boxes = torch.cat(boxes, dim=0)
#         device, dtype = concat_boxes.device, concat_boxes.dtype
#         ids = torch.cat(
#             [
#                 torch.full_like(b[:, :1], i, dtype=dtype, layout=torch.strided, device=device)
#                 for i, b in enumerate(boxes)
#             ],
#             dim=0,
#         )
#         rois = torch.cat([ids, concat_boxes], dim=1)
#         return rois
#
#     def infer_scale(self, feature: Tensor, original_size: List[int]) -> float:
#         # assumption: the scale is of the form 2 ** (-k), with k integer
#         size = feature.shape[-2:]
#         possible_scales = torch.jit.annotate(List[float], [])
#         for s1, s2 in zip(size, original_size):
#             approx_scale = float(s1) / float(s2)
#             scale = 2 ** float(torch.tensor(approx_scale).log2().round())
#             possible_scales.append(scale)
#         assert possible_scales[0] == possible_scales[1]
#         return possible_scales[0]
#
#     def setup_scales(
#         self,
#         features: List[Tensor],
#         image_shapes: List[Tuple[int, int]],
#     ) -> None:
#         assert len(image_shapes) != 0
#         max_x = 0
#         max_y = 0
#         for shape in image_shapes:
#             max_x = max(shape[0], max_x)
#             max_y = max(shape[1], max_y)
#         original_input_shape = (max_x, max_y)
#
#         scales = [self.infer_scale(feat, original_input_shape) for feat in features]
#         # get the levels in the feature map by leveraging the fact that the network always
#         # downsamples by a factor of 2 at each level.
#         lvl_min = -torch.log2(torch.tensor(scales[0], dtype=torch.float32)).item()
#         lvl_max = -torch.log2(torch.tensor(scales[-1], dtype=torch.float32)).item()
#         self.scales = scales
#         self.map_levels = initLevelMapper(int(lvl_min), int(lvl_max))
#
#     def forward(
#         self,
#         x: Dict[str, Tensor],
#         boxes: List[Tensor],
#         image_shapes: List[Tuple[int, int]],    ) -> Tensor:
#         """
#         Arguments:
#             x (OrderedDict[Tensor]): feature maps for each level. They are assumed to have
#                 all the same number of channels, but they can have different sizes.
#             boxes (List[Tensor[N, 4]]): boxes to be used to perform the pooling operation, in
#                 (x1, y1, x2, y2) format and in the image reference size, not the feature map
#                 reference.
#             image_shapes (List[Tuple[height, width]]): the sizes of each image before they
#                 have been fed to a CNN to obtain feature maps. This allows us to infer the
#                 scale factor for each one of the levels to be pooled.
#         Returns:
#             result (Tensor)
#         """
#         x_filtered = []
#         for k, v in x.items():
#             if k in self.featmap_names:
#                 x_filtered.append(v)
#         num_levels = len(x_filtered)
#         rois = self.convert_to_roi_format(boxes)
#         if self.scales is None:
#             self.setup_scales(x_filtered, image_shapes)
#
#         scales = self.scales
#         assert scales is not None
#
#         if num_levels == 1:
#             return roi_align(
#                 x_filtered[0], rois,
#                 output_size=self.output_size,
#                 spatial_scale=scales[0],
#                 sampling_ratio=self.sampling_ratio
#             )
#
#         mapper = self.map_levels
#         assert mapper is not None
#
#         levels = mapper(boxes)
#
#         num_rois = len(rois)
#         num_channels = x_filtered[0].shape[1]
#
#         dtype, device = x_filtered[0].dtype, x_filtered[0].device
#         result = torch.zeros(
#             (num_rois, num_channels,) + self.output_size,
#             dtype=dtype,
#             device=device,
#         )
#
#         tracing_results = []
#         for level, (per_level_feature, scale) in enumerate(zip(x_filtered, scales)):
#             idx_in_level = torch.where(levels == level)[0]
#             rois_per_level = rois[idx_in_level]
#
#             result_idx_in_level = roi_align(
#                 per_level_feature, rois_per_level,
#                 output_size=self.output_size,
#                 spatial_scale=scale, sampling_ratio=self.sampling_ratio)
#
#             if torchvision._is_tracing():
#                 tracing_results.append(result_idx_in_level.to(dtype))
#             else:
#                 # result and result_idx_in_level's dtypes are based on dtypes of different
#                 # elements in x_filtered.  x_filtered contains tensors output by different
#                 # layers.  When autocast is active, it may choose different dtypes for
#                 # different layers' outputs.  Therefore, we defensively match result's dtype
#                 # before copying elements from result_idx_in_level in the following op.
#                 # We need to cast manually (can't rely on autocast to cast for us) because
#                 # the op acts on result in-place, and autocast only affects out-of-place ops.
#                 result[idx_in_level] = result_idx_in_level.to(result.dtype)
#
#         if torchvision._is_tracing():
#             result = _onnx_merge_levels(levels, tracing_results)
#
#         return result



# class RoIHeadsFinalLayer(RoIHeads):
#     """
#     The RoIHeads class for integration of height data in the final NN layer
#     """
#
#     def forward(self,
#                 features,  # type: Dict[str, Tensor]
#                 heights,   # type: Tensor
#                 proposals,  # type: List[Tensor]
#                 image_shapes,  # type: List[Tuple[int, int]]
#                 targets=None  # type: Optional[List[Dict[str, Tensor]]]
#                 ):
#         # type: (...) -> Tuple[List[Dict[str, Tensor]], Dict[str, Tensor]]
#         """
#         Arguments:
#             features (List[Tensor])
#             proposals (List[Tensor[N, 4]])
#             image_shapes (List[Tuple[H, W]])
#             targets (List[Dict])
#         """
#         if targets is not None:
#             for t in targets:
#                 # TODO: https://github.com/pytorch/pytorch/issues/26731
#                 floating_point_types = (torch.float, torch.double, torch.half)
#                 assert t["boxes"].dtype in floating_point_types, 'target boxes must of float type'
#                 assert t["labels"].dtype == torch.int64, 'target labels must of int64 type'
#                 if self.has_keypoint():
#                     assert t["keypoints"].dtype == torch.float32, 'target keypoints must of float type'
#
#         if self.training:
#             proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
#         else:
#             labels = None
#             regression_targets = None
#             matched_idxs = None
#         # Add another layer of features that is the heights
#         #TODO: This will need to be rescaled
#         features["heights"] = heights
#         box_features = self.box_roi_pool(features, proposals, image_shapes)
#         box_features = self.box_head(box_features)
#         class_logits, box_regression = self.box_predictor(box_features)
#
#         result = torch.jit.annotate(List[Dict[str, torch.Tensor]], [])
#         losses = {}
#         if self.training:
#             assert labels is not None and regression_targets is not None
#             loss_classifier, loss_box_reg = fastrcnn_loss(
#                 class_logits, box_regression, labels, regression_targets)
#             losses = {
#                 "loss_classifier": loss_classifier,
#                 "loss_box_reg": loss_box_reg
#             }
#         else:
#             boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
#             num_images = len(boxes)
#             for i in range(num_images):
#                 result.append(
#                     {
#                         "boxes": boxes[i],
#                         "labels": labels[i],
#                         "scores": scores[i],
#                     }
#                 )
#         return result, losses

