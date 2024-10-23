_base_ = "/home/skku/mmdetection/configs/faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py"

# 데이터셋 설정
data_root = "/home/skku/mm_test/data/cargox/"
metainfo = {
   "classes": ("knife1-1", "knife1-2", "knife2-1", "knife2-2", "knife3-1", "knife3-2", "knife4-1", "knife4-2"),
   "palette": [
       (225, 0, 0), (0, 0, 255), (0, 255, 0), (255, 255, 0),
       (0, 255, 255), (255, 0, 255), (255, 165, 0), (0, 0, 128),
   ]
}

# ResNet 모델 대신 ViT 모델로 완전히 교체
model = dict(
   _delete_=True,  # 기존 ResNet 모델 설정 삭제
   type='FasterRCNN',
   data_preprocessor=dict(
       type='DetDataPreprocessor',
       mean=[123.675, 116.28, 103.53],
       std=[58.395, 57.12, 57.375],
       bgr_to_rgb=True,
       pad_size_divisor=32),
   backbone=dict(
       type='mmpretrain.VisionTransformer',
       arch='base',
       img_size=384,
       patch_size=16,
       out_indices=(2, 5, 8, 11),
       drop_rate=0.0,
       drop_path_rate=0.1,
       norm_cfg=dict(type='LN', eps=1e-6),
       out_type='featmap',
       with_cls_token=True,
       final_norm=True,
       init_cfg=dict(
           type='Pretrained',
           checkpoint='/home/skku/mm_test/pretrained/vit-base-p16_in21k-pre-3rdparty_ft-64xb64_in1k-384_20210928-98e8652b.pth'
       )
   ),
   neck=dict(
       type='FPN',
       in_channels=[768, 768, 768, 768],
       out_channels=256,
       num_outs=5
   ),
   rpn_head=dict(
       type='RPNHead',
       in_channels=256,
       feat_channels=256,
       anchor_generator=dict(
           type='AnchorGenerator',
           scales=[8],
           ratios=[0.5, 1.0, 2.0],
           strides=[4, 8, 16, 32, 64]
       ),
       bbox_coder=dict(
           type='DeltaXYWHBBoxCoder',
           target_means=[.0, .0, .0, .0],
           target_stds=[1.0, 1.0, 1.0, 1.0]
       ),
       loss_cls=dict(
           type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0
       ),
       loss_bbox=dict(type='L1Loss', loss_weight=1.0)
   ),
   roi_head=dict(
       type='StandardRoIHead',
       bbox_roi_extractor=dict(
           type='SingleRoIExtractor',
           roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
           out_channels=256,
           featmap_strides=[4, 8, 16, 32]
       ),
       bbox_head=dict(
           type='Shared2FCBBoxHead',
           in_channels=256,
           fc_out_channels=1024,
           roi_feat_size=7,
           num_classes=8,
           bbox_coder=dict(
               type='DeltaXYWHBBoxCoder',
               target_means=[0., 0., 0., 0.],
               target_stds=[0.1, 0.1, 0.2, 0.2]
           ),
           reg_class_agnostic=False,
           loss_cls=dict(
               type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0
           ),
           loss_bbox=dict(type='L1Loss', loss_weight=1.0)
       )
   ),
   train_cfg=dict(
       rpn=dict(
           assigner=dict(
               type='MaxIoUAssigner',
               pos_iou_thr=0.7,
               neg_iou_thr=0.3,
               min_pos_iou=0.3,
               match_low_quality=True,
               ignore_iof_thr=-1),
           sampler=dict(
               type='RandomSampler',
               num=256,
               pos_fraction=0.5,
               neg_pos_ub=-1,
               add_gt_as_proposals=False),
           allowed_border=-1,
           pos_weight=-1,
           debug=False),
       rpn_proposal=dict(
           nms_pre=2000,
           max_per_img=1000,
           nms=dict(type='nms', iou_threshold=0.7),
           min_bbox_size=0),
       rcnn=dict(
           assigner=dict(
               type='MaxIoUAssigner',
               pos_iou_thr=0.5,
               neg_iou_thr=0.5,
               min_pos_iou=0.5,
               match_low_quality=False,
               ignore_iof_thr=-1),
           sampler=dict(
               type='RandomSampler',
               num=512,
               pos_fraction=0.25,
               neg_pos_ub=-1,
               add_gt_as_proposals=True),
           pos_weight=-1,
           debug=False)),
   test_cfg=dict(
       rpn=dict(
           nms_pre=1000,
           max_per_img=1000,
           nms=dict(type='nms', iou_threshold=0.7),
           min_bbox_size=0),
       rcnn=dict(
           score_thr=0.05,
           nms=dict(type='nms', iou_threshold=0.5),
           max_per_img=100))
)

# 데이터 로더 설정
train_dataloader = dict(
   batch_size=1,
   dataset=dict(
       data_root=data_root,
       metainfo=metainfo,
       ann_file='annotations/train.json',
       data_prefix=dict(img='train/'))
)

val_dataloader = dict(
   dataset=dict(
       data_root=data_root,
       metainfo=metainfo,
       ann_file='annotations/val.json',
       data_prefix=dict(img='val/'))
)

test_dataloader = val_dataloader

# 평가 설정
val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = val_evaluator

# 학습률 설정
optim_wrapper = dict(
   _delete_=True,
   type='OptimWrapper',
   optimizer=dict(type='AdamW', lr=0.0001, weight_decay=0.05),
   clip_grad=dict(max_norm=1.0, norm_type=2)
)

# 기본 훅 설정
default_hooks = dict(
   checkpoint=dict(type='CheckpointHook', interval=1)
)
