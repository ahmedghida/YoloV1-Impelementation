import tensorflow as tf 
from tensorflow.keras import backend as K

def IOU(label_box,pred_box):

    #calculate Intersection:
    intersect_mins = K.maximum(pred_box[...,:2],label_box[...,:2])
    intersect_maxes = K.minimum(pred_box[...,2:],label_box[...,2:])
    intersect_wh = K.clip(intersect_maxes - intersect_mins, 0.,None)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    #Calculate Union:
    pred_wh = pred_box[...,2:] - pred_box[...,:2]
    true_wh = label_box[...,2:] - label_box[...,:2]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]
    true_areas = true_wh[..., 0] * true_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = intersect_areas / union_areas

    return iou_scores



def convert2xyxy(Bbox):
        xy_min=(Bbox[...,:2]-Bbox[...,2:])/2
        xy_max=(Bbox[...,:2]+Bbox[...,2:])/2
        Bbox=tf.concat([xy_min, xy_max], axis=-1)
        return Bbox


class Yolo_loss(tf.keras.losses.Loss):
    def __init__(self, S=2, B=1, reduction=tf.keras.losses.Reduction.AUTO, name="yolo_loss"):
        if reduction == 'auto':
            reduction = None  # Replace 'auto' with None
        super(Yolo_loss, self).__init__(reduction=reduction, name=name)
        self.S = S
        self.B = B

    def Calc_Confidance(self,pred_conf,conf_mask):
      no_obj_loss=.5*(1-conf_mask)*K.square(0-pred_conf)
      obj_loss=conf_mask*K.square(1-pred_conf)
      confidance_loss=K.sum(no_obj_loss+obj_loss)
      return confidance_loss


    def call(self, y_true, y_pred,):

      #Extract y_true Info:
      label_class = K.expand_dims(y_true[...,0],axis=-1)
      label_box = y_true[..., 1:5]
      label_conf = K.expand_dims(y_true[...,5],axis=-1)

      #Extract y_pred Info:
      pred_class=y_pred[...,:self.B]
      pred_box=y_pred[...,self.B:(self.B+(self.B*4))]
      pred_conf=y_pred[...,(self.B+(self.B*4)):]

      #Reshaping:
      _label_box = K.reshape(label_box*448, [-1, self.S, self.S,1, 4])
      _predict_box = K.reshape(pred_box*448, [-1, self.S, self.S,self.B, 4])

      #convert from(xcenter,ycenter,w,h) to (xmin,ymin,xmax,ymax)
      xyxy_label_box=convert2xyxy(_label_box)
      xyxy_predict_box=convert2xyxy(_predict_box)

      #Calculate Jaccard Index:
      overlaps=IOU(xyxy_label_box,xyxy_predict_box)
      best_box = K.max(overlaps, axis=3, keepdims=True)
      box_mask = K.cast(overlaps >= best_box, K.dtype(overlaps))
      conf_mask=box_mask*label_conf

      num_element=K.shape(label_class)[0]
      num_element = tf.cast(num_element, tf.float32)
      #Calculate Conf_loss
      Conf_loss=self.Calc_Confidance(pred_conf,conf_mask)/num_element

      #Calculate Class_loss

      class_loss = K.sum(label_conf*K.binary_crossentropy(label_class, pred_class))/num_element
      #class_loss= K.sum(label_conf * K.square(label_class - pred_class))/num_element


      #Calculate Coordinate_loss:
      conf_mask=K.expand_dims(conf_mask,axis=3)
      coor_loss=5*conf_mask*K.square((_label_box[...,:2] - _predict_box[...,:2])/448)
      coor_loss+=5*conf_mask*K.square((K.sqrt(_label_box[...,2:]) - K.sqrt(_predict_box[...,2:]))/448)
      coor_loss=K.sum(coor_loss)/num_element
    
      total_loss=coor_loss+class_loss+Conf_loss
      
      return total_loss,coor_loss,class_loss,Conf_loss