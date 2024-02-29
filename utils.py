import tensorflow as tf
import numpy as np

from loss import convert2xyxy
from tensorflow.keras import backend as K


def format_for_nms(preds,S=4,B=2):
  '''
  preds format and shape : [batch_size,grid_size,grid_size,(n_class+confidance+ 4 ptr for localization)*boxes per grid]
  4 ptr for localization--> xcenter ,ycenter ,w ,h
  B--> is predictions per grid
  S--> is Grid size

  output_format and shape : [batch_size,grid_size,grid_size,boxes per grid, n_class+confidance+ 4 ptr for localization]
  the last axis xmin ymin xmax ymax confidnace class_prob
  '''


  pred_class=preds[...,:B]
  pred_box=preds[...,B:(B+(B*4))]
  pred_conf=preds[...,(B+(B*4)):]

  pred_class=K.reshape(pred_class,[-1,S,S,B,1])
  pred_conf=K.reshape(pred_conf,[-1,S,S,B,1])
  pred_box=K.reshape(pred_box*448,[-1,S,S,B,4])
  xyxy_predict_box=convert2xyxy(pred_box)
  concatenated_tensor = tf.concat([xyxy_predict_box, pred_conf, pred_class], axis=-1)
  return  concatenated_tensor



def nms(pred_boxes, iou_thr=0.7, eps=1e-6):
    """Non-Maximum Suppression
    Args:
        pred_boxes (np.ndarray dtype=np.float32): [x_min, y_min, x_max, y_max, confidence, class_idx]
        iou_thr (float): IoU Threshold (Default: 0.7)
        eps (float): Epsilon value for prevent zero division (Default:1e-6)

    Returns:
        np.ndarray dtype=np.float32: Non-Maximum Suppressed prediction boxes
    """
    if len(pred_boxes) == 0:
        return np.array([], dtype=np.float32)

    x_min, y_min = pred_boxes[:,0], pred_boxes[:,1]
    x_max, y_max = pred_boxes[:,2], pred_boxes[:,3]
    width = np.maximum(x_max - x_min, 0.)
    height = np.maximum(y_max - y_min, 0.)
    area = width * height

    selected_idx_list = list()
    confidence = pred_boxes[:, 4]
    idxs_sorted = np.argsort(confidence)  # Sort in ascending order
    while len(idxs_sorted) > 0:
        max_confidence_idx = len(idxs_sorted) - 1
        non_selected_idxs = idxs_sorted[:max_confidence_idx]
        selected_idx = idxs_sorted[max_confidence_idx]
        selected_idx_list.append(selected_idx)

        inter_xmin = np.maximum(x_min[selected_idx], x_min[non_selected_idxs])
        inter_ymin = np.maximum(y_min[selected_idx], y_min[non_selected_idxs])
        inter_xmax = np.minimum(x_max[selected_idx], x_max[non_selected_idxs])
        inter_ymax = np.minimum(y_max[selected_idx], y_max[non_selected_idxs])
        inter_w = np.maximum(inter_xmax - inter_xmin, 0.)
        inter_h = np.maximum(inter_ymax - inter_ymin, 0.)
        inter_area = inter_w * inter_h

        union = (area[selected_idx] + area[non_selected_idxs]) - inter_area + eps
        iou = inter_area / union
        idxs_sorted = np.delete(idxs_sorted, np.concatenate(([max_confidence_idx], np.where(iou >= iou_thr)[0])))
    return pred_boxes[selected_idx_list]


def box_postp2use(pred_boxes, nms_iou_thr=0.7, conf_thr=0.5):
    """Postprocess prediction boxes to use
    
    * Non-Maximum Suppression
    * Filter boxes with Confidence Score
    
    Args:
      pred_boxes (np.ndarray dtype=np.float32): pred boxes postprocessed by yolo_output2boxes. shape: [cfg.cell_size * cfg.cell_size *cfg.boxes_per_cell, 6]
      nms_iou_thr (float): Non-Maximum Suppression IoU Threshold
      conf_thr (float): Confidence Score Threshold
    
    Returns:
      np.ndarray (dtype=np.float32)
    """
    boxes_nms = nms(pred_boxes=pred_boxes, iou_thr=nms_iou_thr)
    boxes_conf_filtered = boxes_nms[boxes_nms[:, 4] >= conf_thr]
    return boxes_conf_filtered

def detect(model,image):
  preds=model.predict(np.expand_dims(image,axis=0))
  preds=format_for_nms(preds)
  preds=np.squeeze(preds,axis=0)
  preds=preds.reshape(32,6)
  result=box_postp2use(preds)
  plt.imshow(image)
  for i in range(result.shape[0]):
    xmin,ymin,xmax,ymax=result[i,:4]
    w=xmax-xmin
    h=ymax-ymin
    confidences=result[i,4]
    clas=result[i,5]
    if clas>.6:
      clas="Licence"
    rect=patches.Rectangle((xmin, ymin),w,h,linewidth=2, edgecolor='r', facecolor='none')
    plt.gca().add_patch(rect)

    text = f"{clas}: {confidences:.2f}"
    text_bbox = patches.Rectangle((xmin, ymin - 15), w, 15,linewidth=1, edgecolor='r', facecolor='r')
    plt.gca().add_patch(text_bbox)

    plt.text(xmin+10, ymin - 15, text, color='white',fontsize=12, ha='center', va='center')

plt.show()



