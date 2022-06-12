import colorsys
import os
import time

import numpy as np
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
  except RuntimeError as e:
    print(e)
'''
import os
os.environ['CUDA_VISIBLE_DEVICES']="1"#"0,1,2,3,4,5,6,7"
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
'''
from PIL import ImageDraw, ImageFont
from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model


from tensorflow.keras.layers import (BatchNormalization, Concatenate, Input,
                                     Lambda, LeakyReLU, MaxPooling2D,
                                     UpSampling2D, ZeroPadding2D)

from functools import reduce
from functools import wraps
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import (Add, BatchNormalization, Concatenate,
                                     Conv2D, Layer, ZeroPadding2D)
from tensorflow.keras.regularizers import l2
import math
from functools import partial
import tensorflow as tf
from PIL import Image
import tensorflow as tf



###############################################################################

#####  定義函式  #####
def compose(*funcs):
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')


##########
#  主程式
##########

class Mish(Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)
        self.supports_masking = True

    def call(self, inputs):
        return inputs * K.tanh(K.softplus(inputs))

    def get_config(self):
        config = super(Mish, self).get_config()
        return config

    def compute_output_shape(self, input_shape):
        return input_shape

#------------------------------------------------------#
#   單次卷積DarknetConv2D
#   如果步長為2則自己設定padding方式。
#------------------------------------------------------#
@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    darknet_conv_kwargs = {'kernel_initializer' : RandomNormal(stddev=0.02), 'kernel_regularizer' : l2(kwargs.get('weight_decay', 5e-4))}
    darknet_conv_kwargs['padding'] = 'valid' if kwargs.get('strides')==(2, 2) else 'same'   
    try:
        del kwargs['weight_decay']
    except:
        pass
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)



#####  定義函式  #####
#---------------------------------------------------#
#   將預測值的每個特徵層調成真實值
#---------------------------------------------------#
def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    #------------------------------------------#
    #   grid_shape指的是特征层的高和宽
    #------------------------------------------#
    grid_shape = K.shape(feats)[1:3]
    #--------------------------------------------------------------------#
    #   获得各个特征点的坐标信息。生成的shape为(13, 13, num_anchors, 2)
    #--------------------------------------------------------------------#
    grid_x  = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y  = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid    = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))
    #---------------------------------------------------------------#
    #   将先验框进行拓展，生成的shape为(13, 13, num_anchors, 2)
    #---------------------------------------------------------------#
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])

    #---------------------------------------------------#
    #   将预测结果调整成(batch_size,13,13,3,85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心宽高的调整参数
    #   1代表的是框的置信度
    #   80代表的是种类的置信度
    #---------------------------------------------------#
    feats           = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    #------------------------------------------#
    #   对先验框进行解码，并进行归一化
    #------------------------------------------#
    box_xy          = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh          = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    #------------------------------------------#
    #   获得预测框的置信度
    #------------------------------------------#
    box_confidence  = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])
    
    #---------------------------------------------------------------------#
    #   在计算loss的时候返回grid, feats, box_xy, box_wh
    #   在预测的时候返回box_xy, box_wh, box_confidence, box_class_probs
    #---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


##########
#  主程式
##########
def box_ciou(b1, b2):
    """
    輸入為：
    ----------
    b1: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh
    b2: tensor, shape=(batch, feat_w, feat_h, anchor_num, 4), xywh

    返回為：
    -------
    ciou: tensor, shape=(batch, feat_w, feat_h, anchor_num, 1)
    """
    #-----------------------------------------------------------#
    #   求出預測框左上角右下角
    #   b1_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b1_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    #-----------------------------------------------------------#
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh/2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half
    #-----------------------------------------------------------#
    #   求出真實框左上角右下角
    #   b2_mins     (batch, feat_w, feat_h, anchor_num, 2)
    #   b2_maxes    (batch, feat_w, feat_h, anchor_num, 2)
    #-----------------------------------------------------------#
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh/2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    #-----------------------------------------------------------#
    #   求真實框和預測框所有的iou
    #   iou         (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    intersect_mins = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    union_area = b1_area + b2_area - intersect_area
    iou = intersect_area / K.maximum(union_area, K.epsilon())

    #-----------------------------------------------------------#
    #   計算中心的差距
    #   center_distance (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    center_distance = K.sum(K.square(b1_xy - b2_xy), axis=-1)
    enclose_mins = K.minimum(b1_mins, b2_mins)
    enclose_maxes = K.maximum(b1_maxes, b2_maxes)
    enclose_wh = K.maximum(enclose_maxes - enclose_mins, 0.0)
    #-----------------------------------------------------------#
    #   計算對角線距離
    #   enclose_diagonal (batch, feat_w, feat_h, anchor_num)
    #-----------------------------------------------------------#
    enclose_diagonal = K.sum(K.square(enclose_wh), axis=-1)
    ciou = iou - 1.0 * (center_distance) / K.maximum(enclose_diagonal ,K.epsilon())
    
    v = 4 * K.square(tf.math.atan2(b1_wh[..., 0], K.maximum(b1_wh[..., 1], K.epsilon())) - tf.math.atan2(b2_wh[..., 0], K.maximum(b2_wh[..., 1],K.epsilon()))) / (math.pi * math.pi)
    alpha = v /  K.maximum((1.0 - iou + v), K.epsilon())
    ciou = ciou - alpha * v

    ciou = K.expand_dims(ciou, -1)
    return ciou


#---------------------------------------------------#
#   平滑標籤
#---------------------------------------------------#
def _smooth_labels(y_true, label_smoothing):
    num_classes = tf.cast(K.shape(y_true)[-1], dtype=K.floatx())
    label_smoothing = K.constant(label_smoothing, dtype=K.floatx())
    return y_true * (1.0 - label_smoothing) + label_smoothing / num_classes


#---------------------------------------------------#
#   用於計算每個預測框與真實框的iou
#---------------------------------------------------#
def box_iou(b1, b2):
    #---------------------------------------------------#
    #   num_anchor,1,4
    #   計算左上角的坐標和右下角的坐標
    #---------------------------------------------------#
    b1          = K.expand_dims(b1, -2)
    b1_xy       = b1[..., :2]
    b1_wh       = b1[..., 2:4]
    b1_wh_half  = b1_wh/2.
    b1_mins     = b1_xy - b1_wh_half
    b1_maxes    = b1_xy + b1_wh_half

    #---------------------------------------------------#
    #   1,n,4
    #   計算左上角和右下角的坐標
    #---------------------------------------------------#
    b2          = K.expand_dims(b2, 0)
    b2_xy       = b2[..., :2]
    b2_wh       = b2[..., 2:4]
    b2_wh_half  = b2_wh/2.
    b2_mins     = b2_xy - b2_wh_half
    b2_maxes    = b2_xy + b2_wh_half

    #---------------------------------------------------#
    #   計算重合面積
    #---------------------------------------------------#
    intersect_mins  = K.maximum(b1_mins, b2_mins)
    intersect_maxes = K.minimum(b1_maxes, b2_maxes)
    intersect_wh    = K.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area         = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area         = b2_wh[..., 0] * b2_wh[..., 1]
    iou             = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


#---------------------------------------------------#
#   loss值計算
#---------------------------------------------------#
def yolo_loss(
    args, 
    input_shape, 
    anchors, 
    anchors_mask, 
    num_classes, 
    ignore_thresh   = 0.5,
    balance         = [0.4, 1.0, 4], 
    box_ratio       = 0.05, 
    obj_ratio       = 1, 
    cls_ratio       = 0.5 / 4, 
    label_smoothing = 0.1, 
    print_loss      = False
):
    num_layers = len(anchors_mask)
    #---------------------------------------------------------------------------------------------------#
    #   將預測結果和實際ground truth分開，args是[*model_body.output, *y_true]
    #   y_true是一個列表，包含三個特徵層，shape分別為:
    #   (m,13,13,3,85)
    #   (m,26,26,3,85)
    #   (m,52,52,3,85)
    #   yolo_outputs是一個列表，包含三個特徵層，shape分別為:
    #   (m,13,13,3,85)
    #   (m,26,26,3,85)
    #   (m,52,52,3,85)
    #---------------------------------------------------------------------------------------------------#
    y_true          = args[num_layers:]
    yolo_outputs    = args[:num_layers]

    #-----------------------------------------------------------#
    #   得到input_shpae為416,416 
    #-----------------------------------------------------------#
    input_shape = K.cast(input_shape, K.dtype(y_true[0]))

    #-----------------------------------------------------------#
    #   取出每一張圖片
    #   m的值就是batch_size
    #-----------------------------------------------------------#
    m = K.shape(yolo_outputs[0])[0]

    loss    = 0
    #---------------------------------------------------------------------------------------------------#
    #   y_true是一個列表，包含三個特徵層，shape分別為(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    #   yolo_outputs是一個列表，包含三個特徵層，shape分別為(m,13,13,3,85),(m,26,26,3,85),(m,52,52,3,85)。
    #---------------------------------------------------------------------------------------------------#
    for l in range(num_layers):
        #-----------------------------------------------------------#
        #   以第一個特徵層(m,13,13,3,85)為例子
        #   取出該特徵層中存在目標的點的位置。 (m,13,13,3,1)
        #-----------------------------------------------------------#
        object_mask         = y_true[l][..., 4:5]
        #-----------------------------------------------------------#
        #   取出其對應的種類(m,13,13,3,80)
        #-----------------------------------------------------------#
        true_class_probs    = y_true[l][..., 5:]
        if label_smoothing:
            true_class_probs = _smooth_labels(true_class_probs, label_smoothing)

        #-----------------------------------------------------------#
        #   將yolo_outputs的特徵層輸出進行處理、獲得四個返回值
        #   其中：
        #   grid        (13,13,1,2) 網格坐標
        #   raw_pred    (m,13,13,3,85) 尚未處理的預測結果
        #   pred_xy     (m,13,13,3,2) 解碼後的中心坐標
        #   pred_wh     (m,13,13,3,2) 解碼後的寬高坐標
        #-----------------------------------------------------------#
        grid, raw_pred, pred_xy, pred_wh = get_anchors_and_decode(yolo_outputs[l],
             anchors[anchors_mask[l]], num_classes, input_shape, calc_loss=True)
        
        #-----------------------------------------------------------#
        #   pred_box是解碼後的預測的box的位置
        #   (m,13,13,3,4)
        #-----------------------------------------------------------#
        pred_box = K.concatenate([pred_xy, pred_wh])

        #-----------------------------------------------------------#
        #   找到負樣本群組，第一步是創建一個數組，[]
        #-----------------------------------------------------------#
        ignore_mask = tf.TensorArray(K.dtype(y_true[0]), size=1, dynamic_size=True)
        object_mask_bool = K.cast(object_mask, 'bool')
        
        #-----------------------------------------------------------#
        #   對每一張圖片計算ignore_mask
        #-----------------------------------------------------------#
        def loop_body(b, ignore_mask):
            #-----------------------------------------------------------#
            #   取出n個真實框：n,4
            #-----------------------------------------------------------#
            true_box = tf.boolean_mask(y_true[l][b,...,0:4], object_mask_bool[b,...,0])
            #-----------------------------------------------------------#
            #   計算預測框與真實框的iou
            #   pred_box    13,13,3,4 預測框的坐標
            #   true_box    n,4 真實框的坐標
            #   iou         13,13,3,n 預測框和真實框的iou
            #-----------------------------------------------------------#
            iou = box_iou(pred_box[b], true_box)

            #-----------------------------------------------------------#
            #   best_iou    13,13,3 每個特徵點與真實框的最大重合程度
            #-----------------------------------------------------------#
            best_iou = K.max(iou, axis=-1)

            #-----------------------------------------------------------#
            #   判斷預測框和真實框的最大iou小於ignore_thresh
            #   則認為該預測框沒有與之對應的真實框
            #   該操作的目的是：
            #   忽略預測結果與真實框非常對應特徵點，因為這些框已經比較準了
            #   不適合當作負樣本，所以忽略掉。
            #-----------------------------------------------------------#
            ignore_mask = ignore_mask.write(b, K.cast(best_iou<ignore_thresh, K.dtype(true_box)))
            return b+1, ignore_mask

        #-----------------------------------------------------------#
        #   在這個地方進行一個循環、循環是對每一張圖片進行的
        #-----------------------------------------------------------#
        _, ignore_mask = tf.while_loop(lambda b,*args: b<m, loop_body, [0, ignore_mask])

        #-----------------------------------------------------------#
        #   ignore_mask用於提取出作為負樣本的特徵點
        #   (m,13,13,3)
        #-----------------------------------------------------------#
        ignore_mask = ignore_mask.stack()
        #   (m,13,13,3,1)
        ignore_mask = K.expand_dims(ignore_mask, -1)

        #-----------------------------------------------------------#
        #   真實框越大，比重越小，小框的比重更大。
        #   使用iou損失時，大中小目標的回歸損失不存在比例失衡問題，故棄用
        #-----------------------------------------------------------#
        box_loss_scale = 2 - y_true[l][...,2:3]*y_true[l][...,3:4]

        #-----------------------------------------------------------#
        #   計算Ciou loss
        #-----------------------------------------------------------#
        raw_true_box    = y_true[l][...,0:4]
        ciou            = box_ciou(pred_box, raw_true_box)
        ciou_loss       = object_mask * (1 - ciou)
        location_loss   = K.sum(ciou_loss)
        
        #------------------------------------------------------------------------------#
        #   如果該位置本來有框，那麼計算1與置信度的交叉熵
        #   如果該位置本來沒有框，那麼計算0與置信度的交叉熵
        #   在這其中會忽略一部分樣本，這些被忽略的樣本滿足條件best_iou<ignore_thresh
        #   該操作的目的是：
        #   忽略預測結果與真實框非常對應特徵點，因為這些框已經比較準了
        #   不適合當作負樣本，所以忽略掉。
        #------------------------------------------------------------------------------#
        confidence_loss = object_mask * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) + \
                    (1 - object_mask) * K.binary_crossentropy(object_mask, raw_pred[...,4:5], from_logits=True) * ignore_mask
        
        class_loss      = object_mask * K.binary_crossentropy(true_class_probs, raw_pred[...,5:], from_logits=True)
        
        #-----------------------------------------------------------#
        #   計算正樣本數量
        #-----------------------------------------------------------#
        num_pos         = tf.maximum(K.sum(K.cast(object_mask, tf.float32)), 1)
        num_neg         = tf.maximum(K.sum(K.cast((1 - object_mask) * ignore_mask, tf.float32)), 1)

        #-----------------------------------------------------------#
        #   將所有損失求和
        #-----------------------------------------------------------#
        location_loss   = location_loss * box_ratio / num_pos
        confidence_loss = K.sum(confidence_loss) * balance[l] * obj_ratio / (num_pos + num_neg)
        class_loss      = K.sum(class_loss) * cls_ratio / num_pos / num_classes

        loss            += location_loss + confidence_loss + class_loss
        if print_loss:
            loss = tf.Print(loss, [loss, location_loss, confidence_loss, class_loss, tf.shape(ignore_mask)], summarize=100, message='loss: ')
    return loss


#---------------------------------------------------#
#   卷積塊 -> 卷積 + 標準化 + 激活函數
#   DarknetConv2D + BatchNormalization + Mish
#---------------------------------------------------#
def DarknetConv2D_BN_Mish(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        Mish())


#--------------------------------------------------------------------#
#   CSPdarknet的結構塊
#   首先利用ZeroPadding2D和一個步長為2x2的捲積塊進行高和寬的壓縮
#   然後建立一個大的殘差邊shortconv、這個大殘差邊繞過了很多的殘差結構
#   主幹部分會對num_blocks進行循環，循環內部是殘差結構。
#   對於整個CSPdarknet的結構塊，就是一個大殘差塊+內部多個小殘差塊
#--------------------------------------------------------------------#
def resblock_body(x, num_filters, num_blocks, all_narrow=True, weight_decay=5e-4):
    #----------------------------------------------------------------#
    #   利用ZeroPadding2D和一個步長為2x2的捲積塊進行高和寬的壓縮
    #----------------------------------------------------------------#
    preconv1 = ZeroPadding2D(((1,0),(1,0)))(x)
    preconv1 = DarknetConv2D_BN_Mish(num_filters, (3,3), strides=(2,2), weight_decay=weight_decay)(preconv1)

    #--------------------------------------------------------------------#
    #   然後建立一個大的殘差邊shortconv、這個大殘差邊繞過了很多的殘差結構
    #--------------------------------------------------------------------#
    shortconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1), weight_decay=weight_decay)(preconv1)

    #----------------------------------------------------------------#
    #   主幹部分會對num_blocks進行循環，循環內部是殘差結構。
    #----------------------------------------------------------------#
    mainconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1), weight_decay=weight_decay)(preconv1)
    for i in range(num_blocks):
        y = compose(
                DarknetConv2D_BN_Mish(num_filters//2, (1,1), weight_decay=weight_decay),
                DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (3,3), weight_decay=weight_decay))(mainconv)
        mainconv = Add()([mainconv,y])
    postconv = DarknetConv2D_BN_Mish(num_filters//2 if all_narrow else num_filters, (1,1), weight_decay=weight_decay)(mainconv)

    #----------------------------------------------------------------#
    #   將大殘差邊再堆疊回來
    #----------------------------------------------------------------#
    route = Concatenate()([postconv, shortconv])

    # 最後對通道數進行整合
    return DarknetConv2D_BN_Mish(num_filters, (1,1))(route)


#---------------------------------------------------#
#   CSPdarknet53 的主體部分
#   輸入為一張416x416x3的圖片
#   輸出為三個有效特徵層
#---------------------------------------------------#
def darknet_body(x, weight_decay=5e-4):
    x = DarknetConv2D_BN_Mish(32, (3,3), weight_decay=weight_decay)(x)
    x = resblock_body(x, 64, 1, False, weight_decay=weight_decay)
    x = resblock_body(x, 128, 2, weight_decay=weight_decay)
    x = resblock_body(x, 256, 8, weight_decay=weight_decay)
    feat1 = x
    x = resblock_body(x, 512, 8, weight_decay=weight_decay)
    feat2 = x
    x = resblock_body(x, 1024, 4, weight_decay=weight_decay)
    feat3 = x
    return feat1,feat2,feat3


##########
#  主程式
##########

#---------------------------------------------------#
#   卷積塊 -> 卷積 + 標準化 + 激活函數
#   DarknetConv2D + BatchNormalization + LeakyReLU
#---------------------------------------------------#
def DarknetConv2D_BN_Leaky(*args, **kwargs):
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose( 
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))

#---------------------------------------------------#
#   進行五次卷積
#---------------------------------------------------#
def make_five_convs(x, num_filters, weight_decay=5e-4):
    # 五次卷積
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1), weight_decay=weight_decay)(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3), weight_decay=weight_decay)(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1), weight_decay=weight_decay)(x)
    x = DarknetConv2D_BN_Leaky(num_filters*2, (3,3), weight_decay=weight_decay)(x)
    x = DarknetConv2D_BN_Leaky(num_filters, (1,1), weight_decay=weight_decay)(x)
    return x

#---------------------------------------------------#
#   Panet網絡的構建，並且獲得預測結果
#---------------------------------------------------#
def yolo_body(input_shape, anchors_mask, num_classes, weight_decay=5e-4):
    inputs      = Input(input_shape)
    #---------------------------------------------------#   
    #   生成CSPdarknet53的主幹模型
    #   獲得三個有效特徵層，他們的shape分別是：
    #   52,52,256
    #   26,26,512
    #   13,13,1024
    #---------------------------------------------------#
    feat1,feat2,feat3 = darknet_body(inputs)

    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,2048 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    P5 = DarknetConv2D_BN_Leaky(512, (1,1), weight_decay=weight_decay)(feat3)
    P5 = DarknetConv2D_BN_Leaky(1024, (3,3), weight_decay=weight_decay)(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1,1), weight_decay=weight_decay)(P5)
    # 使用了SPP結構，即不同尺度的最大池化後堆疊。
    maxpool1 = MaxPooling2D(pool_size=(13,13), strides=(1,1), padding='same')(P5)
    maxpool2 = MaxPooling2D(pool_size=(9,9), strides=(1,1), padding='same')(P5)
    maxpool3 = MaxPooling2D(pool_size=(5,5), strides=(1,1), padding='same')(P5)
    P5 = Concatenate()([maxpool1, maxpool2, maxpool3, P5])
    P5 = DarknetConv2D_BN_Leaky(512, (1,1), weight_decay=weight_decay)(P5)
    P5 = DarknetConv2D_BN_Leaky(1024, (3,3), weight_decay=weight_decay)(P5)
    P5 = DarknetConv2D_BN_Leaky(512, (1,1), weight_decay=weight_decay)(P5)

    # 13,13,512 -> 13,13,256 -> 26,26,256
    P5_upsample = compose(DarknetConv2D_BN_Leaky(256, (1,1), weight_decay=weight_decay), UpSampling2D(2))(P5)
    # 26,26,512 -> 26,26,256
    P4 = DarknetConv2D_BN_Leaky(256, (1,1), weight_decay=weight_decay)(feat2)
    # 26,26,256 + 26,26,256 -> 26,26,512
    P4 = Concatenate()([P4, P5_upsample])
    
    # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    P4 = make_five_convs(P4, 256, weight_decay=weight_decay)

    # 26,26,256 -> 26,26,128 -> 52,52,128
    P4_upsample = compose(DarknetConv2D_BN_Leaky(128, (1,1), weight_decay=weight_decay), UpSampling2D(2))(P4)
    # 52,52,256 -> 52,52,128
    P3 = DarknetConv2D_BN_Leaky(128, (1,1), weight_decay=weight_decay)(feat1)
    # 52,52,128 + 52,52,128 -> 52,52,256
    P3 = Concatenate()([P3, P4_upsample])

    # 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
    P3 = make_five_convs(P3, 128, weight_decay=weight_decay)
    
    #---------------------------------------------------#
    #   第三個特徵層
    #   y3=(batch_size,52,52,3,85)
    #---------------------------------------------------#
    P3_output = DarknetConv2D_BN_Leaky(256, (3,3), weight_decay=weight_decay)(P3)
    P3_output = DarknetConv2D(len(anchors_mask[0])*(num_classes+5), (1,1), weight_decay=weight_decay)(P3_output)

    # 52,52,128 -> 26,26,256
    P3_downsample = ZeroPadding2D(((1,0),(1,0)))(P3)
    P3_downsample = DarknetConv2D_BN_Leaky(256, (3,3), strides=(2,2), weight_decay=weight_decay)(P3_downsample)
    # 26,26,256 + 26,26,256 -> 26,26,512
    P4 = Concatenate()([P3_downsample, P4])
    # 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
    P4 = make_five_convs(P4, 256, weight_decay=weight_decay)
    
    #---------------------------------------------------#
    #   第二個特徵層
    #   y2=(batch_size,26,26,3,85)
    #---------------------------------------------------#
    P4_output = DarknetConv2D_BN_Leaky(512, (3,3), weight_decay=weight_decay)(P4)
    P4_output = DarknetConv2D(len(anchors_mask[1])*(num_classes+5), (1,1), weight_decay=weight_decay)(P4_output)
    
    # 26,26,256 -> 13,13,512
    P4_downsample = ZeroPadding2D(((1,0),(1,0)))(P4)
    P4_downsample = DarknetConv2D_BN_Leaky(512, (3,3), strides=(2,2), weight_decay=weight_decay)(P4_downsample)
    # 13,13,512 + 13,13,512 -> 13,13,1024
    P5 = Concatenate()([P4_downsample, P5])
    # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
    P5 = make_five_convs(P5, 512, weight_decay=weight_decay)
    
    #---------------------------------------------------#
    #   第一個特徵層
    #   y1=(batch_size,13,13,3,85)
    #---------------------------------------------------#
    P5_output = DarknetConv2D_BN_Leaky(1024, (3,3), weight_decay=weight_decay)(P5)
    P5_output = DarknetConv2D(len(anchors_mask[2])*(num_classes+5), (1,1), weight_decay=weight_decay)(P5_output)

    return Model(inputs, [P5_output, P4_output, P3_output])



#####  定義函式  #####
#---------------------------------------------------------#
#   將圖像轉換成RGB圖像，防止灰度圖在預測時報錯。
#   代碼僅僅支持RGB圖像的預測，所有其它類型的圖像都會轉化成RGB
#---------------------------------------------------------#
def cvtColor(image):
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image 
    else:
        image = image.convert('RGB')
        return image 

#---------------------------------------------------#
#   對輸入圖像進行resize
#---------------------------------------------------#
def resize_image(image, size, letterbox_image):
    iw, ih  = image.size
    w, h    = size
    if letterbox_image:
        scale   = min(w/iw, h/ih)
        nw      = int(iw*scale)
        nh      = int(ih*scale)

        image   = image.resize((nw,nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (128,128,128))
        new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    else:
        new_image = image.resize((w, h), Image.BICUBIC)
    return new_image

#---------------------------------------------------#
#   獲得類
#---------------------------------------------------#
def get_classes(classes_path):
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names, len(class_names)

#---------------------------------------------------#
#   獲得先驗框
#---------------------------------------------------#
def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path, encoding='utf-8') as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    anchors = np.array(anchors).reshape(-1, 2)
    return anchors, len(anchors)

def preprocess_input(image):
    image /= 255.0
    return image



#####  定義函式  #####
#---------------------------------------------------#
#   對box進行調整，使其符合真實圖片的樣子
#---------------------------------------------------#
def yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image):
    #-----------------------------------------------------------------#
    #   把y軸放前面是因為方便預測框和圖像的寬高進行相乘
    #-----------------------------------------------------------------#
    box_yx = box_xy[..., ::-1]
    box_hw = box_wh[..., ::-1]
    input_shape = K.cast(input_shape, K.dtype(box_yx))
    image_shape = K.cast(image_shape, K.dtype(box_yx))

    if letterbox_image:
        #-----------------------------------------------------------------#
        #   這裡求出來的offset是圖像有效區域相對於圖像左上角的偏移情況
        #   new_shape指的是寬高縮放情況
        #-----------------------------------------------------------------#
        new_shape = K.round(image_shape * K.min(input_shape/image_shape))
        offset  = (input_shape - new_shape)/2./input_shape
        scale   = input_shape/new_shape

        box_yx  = (box_yx - offset) * scale
        box_hw *= scale

    box_mins    = box_yx - (box_hw / 2.)
    box_maxes   = box_yx + (box_hw / 2.)
    boxes  = K.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]])
    boxes *= K.concatenate([image_shape, image_shape])
    return boxes

#---------------------------------------------------#
#   將預測值的每個特徵層調成真實值
#---------------------------------------------------#
def get_anchors_and_decode(feats, anchors, num_classes, input_shape, calc_loss=False):
    num_anchors = len(anchors)
    #------------------------------------------#
    #   grid_shape指的是特徵層的高和寬
    #------------------------------------------#
    grid_shape = K.shape(feats)[1:3]
    #--------------------------------------------------------------------#
    #   獲得各個特徵點的坐標信息。生成的shape為(13, 13, num_anchors, 2)
    #--------------------------------------------------------------------#
    grid_x  = K.tile(K.reshape(K.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
    grid_y  = K.tile(K.reshape(K.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
    grid    = K.cast(K.concatenate([grid_x, grid_y]), K.dtype(feats))
    #---------------------------------------------------------------#
    #   將先驗框進行拓展，生成的shape為(13, 13, num_anchors, 2)
    #---------------------------------------------------------------#
    anchors_tensor = K.reshape(K.constant(anchors), [1, 1, num_anchors, 2])
    anchors_tensor = K.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1])

    #---------------------------------------------------#
    #   將預測結果調整成(batch_size,13,13,3,85)
    #   85可拆分成4 + 1 + 80
    #   4代表的是中心寬高的調整參數
    #   1代表的是框的置信度
    #   80代表的是種類的置信度
    #---------------------------------------------------#
    feats           = K.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
    #------------------------------------------#
    #   對先驗框進行解碼，並進行歸一化
    #------------------------------------------#
    box_xy          = (K.sigmoid(feats[..., :2]) + grid) / K.cast(grid_shape[::-1], K.dtype(feats))
    box_wh          = K.exp(feats[..., 2:4]) * anchors_tensor / K.cast(input_shape[::-1], K.dtype(feats))
    #------------------------------------------#
    #   獲得預測框的置信度
    #------------------------------------------#
    box_confidence  = K.sigmoid(feats[..., 4:5])
    box_class_probs = K.sigmoid(feats[..., 5:])
    
    #---------------------------------------------------------------------#
    #   在計算loss的時候返回grid, feats, box_xy, box_wh
    #   在預測的時候返回box_xy, box_wh, box_confidence, box_class_probs
    #---------------------------------------------------------------------#
    if calc_loss == True:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs

#---------------------------------------------------#
#   圖片預測
#---------------------------------------------------#
def DecodeBox(outputs,
            anchors,
            num_classes,
            input_shape,
            #-----------------------------------------------------------#
            #   13x13的特徵層對應的anchor是[116,90],[156,198],[373,326]
            #   26x26的特徵層對應的anchor是[30,61],[62,45],[59,119]
            #   52x52的特徵層對應的anchor是[10,13],[16,30],[33,23]
            #-----------------------------------------------------------#
            anchor_mask     = [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
            max_boxes       = 100,
            confidence      = 0.5,
            nms_iou         = 0.3,
            letterbox_image = True):
    
    image_shape = K.reshape(outputs[-1],[-1])

    box_xy = []
    box_wh = []
    box_confidence  = []
    box_class_probs = []
    for i in range(len(anchor_mask)):
        sub_box_xy, sub_box_wh, sub_box_confidence, sub_box_class_probs = \
            get_anchors_and_decode(outputs[i], anchors[anchor_mask[i]], num_classes, input_shape)
        box_xy.append(K.reshape(sub_box_xy, [-1, 2]))
        box_wh.append(K.reshape(sub_box_wh, [-1, 2]))
        box_confidence.append(K.reshape(sub_box_confidence, [-1, 1]))
        box_class_probs.append(K.reshape(sub_box_class_probs, [-1, num_classes]))
    box_xy          = K.concatenate(box_xy, axis = 0)
    box_wh          = K.concatenate(box_wh, axis = 0)
    box_confidence  = K.concatenate(box_confidence, axis = 0)
    box_class_probs = K.concatenate(box_class_probs, axis = 0)
    
    #------------------------------------------------------------------------------------------------------------#
    #   在圖像傳入網絡預測前會進行letterbox_image給圖像周圍添加灰條，因此生成的box_xy, box_wh是相對於有灰條的圖像的
    #   我們需要對其進行修改，去除灰條的部分。將box_xy、和box_wh調節成y_min,y_max,xmin,xmax
    #   如果沒有使用letterbox_image也需要將歸一化後的box_xy, box_wh調整成相對於原圖大小的
    #------------------------------------------------------------------------------------------------------------#
    boxes       = yolo_correct_boxes(box_xy, box_wh, input_shape, image_shape, letterbox_image)

    box_scores  = box_confidence * box_class_probs

    #-----------------------------------------------------------#
    #   判斷得分是否大於score_threshold
    #-----------------------------------------------------------#
    mask             = box_scores >= confidence
    max_boxes_tensor = K.constant(max_boxes, dtype='int32')
    boxes_out   = []
    scores_out  = []
    classes_out = []
    for c in range(num_classes):
        #-----------------------------------------------------------#
        #   取出所有box_scores >= score_threshold的框，和成績
        #-----------------------------------------------------------#
        class_boxes      = tf.boolean_mask(boxes, mask[:, c])
        class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])

        #-----------------------------------------------------------#
        #   非極大抑制
        #   保留一定區域內得分最大的框
        #-----------------------------------------------------------#
        nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor, iou_threshold=nms_iou)

        #-----------------------------------------------------------#
        #   獲取非極大抑制後的結果
        #   下列三個分別是：框的位置，得分與種類
        #-----------------------------------------------------------#
        class_boxes         = K.gather(class_boxes, nms_index)
        class_box_scores    = K.gather(class_box_scores, nms_index)
        classes             = K.ones_like(class_box_scores, 'int32') * c

        boxes_out.append(class_boxes)
        scores_out.append(class_box_scores)
        classes_out.append(classes)
    boxes_out      = K.concatenate(boxes_out, axis=0)
    scores_out     = K.concatenate(scores_out, axis=0)
    classes_out    = K.concatenate(classes_out, axis=0)

    return boxes_out, scores_out, classes_out


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    def sigmoid(x):
        s = 1 / (1 + np.exp(-x))
        return s
    #---------------------------------------------------#
    #   将预测值的每个特征层调成真实值
    #---------------------------------------------------#
    def get_anchors_and_decode(feats, anchors, num_classes):
        # feats     [batch_size, 13, 13, 3 * (5 + num_classes)]
        # anchors   [3, 2]
        # num_classes 
        # 3
        num_anchors = len(anchors)
        #------------------------------------------#
        #   grid_shape指的是特征层的高和宽
        #   grid_shape [13, 13] 
        #------------------------------------------#
        grid_shape = np.shape(feats)[1:3]
        #--------------------------------------------------------------------#
        #   获得各个特征点的坐标信息。生成的shape为(13, 13, num_anchors, 2)
        #   grid_x [13, 13, 3, 1]
        #   grid_y [13, 13, 3, 1]
        #   grid   [13, 13, 3, 2]
        #--------------------------------------------------------------------#
        grid_x  = np.tile(np.reshape(np.arange(0, stop=grid_shape[1]), [1, -1, 1, 1]), [grid_shape[0], 1, num_anchors, 1])
        grid_y  = np.tile(np.reshape(np.arange(0, stop=grid_shape[0]), [-1, 1, 1, 1]), [1, grid_shape[1], num_anchors, 1])
        grid    = np.concatenate([grid_x, grid_y], -1)
        #---------------------------------------------------------------#
        #   将先验框进行拓展，生成的shape为(13, 13, num_anchors, 2)
        #   [1, 1, 3, 2]
        #   [13, 13, 3, 2]
        #---------------------------------------------------------------#
        anchors_tensor = np.reshape(anchors, [1, 1, num_anchors, 2])
        anchors_tensor = np.tile(anchors_tensor, [grid_shape[0], grid_shape[1], 1, 1]) 

        #---------------------------------------------------#
        #   将预测结果调整成(batch_size,13,13,3,85)
        #   85可拆分成4 + 1 + 80
        #   4代表的是中心宽高的调整参数
        #   1代表的是框的置信度
        #   80代表的是种类的置信度
        #   [batch_size, 13, 13, 3 * (5 + num_classes)]
        #   [batch_size, 13, 13, 3, 5 + num_classes]
        #---------------------------------------------------#
        feats           = np.reshape(feats, [-1, grid_shape[0], grid_shape[1], num_anchors, num_classes + 5])
        #------------------------------------------#
        #   对先验框进行解码，并进行归一化
        #------------------------------------------#
        box_xy          = sigmoid(feats[..., :2]) + grid
        box_wh          = np.exp(feats[..., 2:4]) * anchors_tensor
        #------------------------------------------#
        #   获得预测框的置信度
        #------------------------------------------#
        box_confidence  = sigmoid(feats[..., 4:5])
        box_class_probs = sigmoid(feats[..., 5:])

        box_wh = box_wh / 32
        anchors_tensor = anchors_tensor / 32
        fig = plt.figure()
        ax = fig.add_subplot(121)
        plt.ylim(-2,15)
        plt.xlim(-2,15)
        plt.scatter(grid_x,grid_y)
        plt.scatter(5,5,c='black')
        plt.gca().invert_yaxis()


        anchor_left = grid_x - anchors_tensor/2 
        anchor_top = grid_y - anchors_tensor/2 
        print(np.shape(anchors_tensor))
        print(np.shape(box_xy))
        rect1 = plt.Rectangle([anchor_left[5,5,0,0],anchor_top[5,5,0,1]],anchors_tensor[0,0,0,0],anchors_tensor[0,0,0,1],color="r",fill=False)
        rect2 = plt.Rectangle([anchor_left[5,5,1,0],anchor_top[5,5,1,1]],anchors_tensor[0,0,1,0],anchors_tensor[0,0,1,1],color="r",fill=False)
        rect3 = plt.Rectangle([anchor_left[5,5,2,0],anchor_top[5,5,2,1]],anchors_tensor[0,0,2,0],anchors_tensor[0,0,2,1],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        ax = fig.add_subplot(122)
        plt.ylim(-2,15)
        plt.xlim(-2,15)
        plt.scatter(grid_x,grid_y)
        plt.scatter(5,5,c='black')
        plt.scatter(box_xy[0,5,5,:,0],box_xy[0,5,5,:,1],c='r')
        plt.gca().invert_yaxis()

        pre_left = box_xy[...,0] - box_wh[...,0]/2 
        pre_top = box_xy[...,1] - box_wh[...,1]/2 

        rect1 = plt.Rectangle([pre_left[0,5,5,0],pre_top[0,5,5,0]],box_wh[0,5,5,0,0],box_wh[0,5,5,0,1],color="r",fill=False)
        rect2 = plt.Rectangle([pre_left[0,5,5,1],pre_top[0,5,5,1]],box_wh[0,5,5,1,0],box_wh[0,5,5,1,1],color="r",fill=False)
        rect3 = plt.Rectangle([pre_left[0,5,5,2],pre_top[0,5,5,2]],box_wh[0,5,5,2,0],box_wh[0,5,5,2,1],color="r",fill=False)

        ax.add_patch(rect1)
        ax.add_patch(rect2)
        ax.add_patch(rect3)

        plt.show()
        #
    feat = np.random.normal(0,0.5,[4,13,13,75])
    anchors = [[142, 110],[192, 243],[459, 401]]
    get_anchors_and_decode(feat,anchors,20)




###############################################################################
#  主程式
###############################################################################

class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   使用自己訓練好的模型進行預測一定要修改model_path和classes_path！
        #   model_path指向logs文件夾下的權值文件，classes_path指向model_data下的txt
        #
        #   訓練好後logs文件夾下存在多個權值文件，選擇驗證集損失較低的即可。
        #   驗證集損失較低不代表mAP較高，僅代表該權值在驗證集上泛化性能較好。
        #   如果出現shape不匹配，同時要注意訓練時的model_path和classes_path參數的修改
        #--------------------------------------------------------------------------#
        "model_path"        : 'model_data/ep100-loss24.429-val_loss24.412.h5',
        "classes_path"      : 'model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path代表先驗框對應的txt文件，一般不修改。
        #   anchors_mask用於幫助代碼找到對應的先驗框，一般不修改。
        #---------------------------------------------------------------------#
        "anchors_path"      : 'model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   輸入圖片的大小，必須為32的倍數。
        #---------------------------------------------------------------------#
        "input_shape"       : [416, 416],
        #---------------------------------------------------------------------#
        #   只有得分大於置信度的預測框會被保留下來
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   非極大抑制所用到的nms_iou大小
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   最大框的數量
        #---------------------------------------------------------------------#
        "max_boxes"         : 100,
        #---------------------------------------------------------------------#
        #   該變量用於控制是否使用letterbox_image對輸入圖像進行不失真的resize，
        #   在多次測試後，發現關閉letterbox_image直接resize的效果更好
        #---------------------------------------------------------------------#
        "letterbox_image"   : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化yolo
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   獲得種類和先驗框的數量
        #---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors, self.num_anchors     = get_anchors(self.anchors_path)

        #---------------------------------------------------#
        #   畫框設置不同的顏色
        #---------------------------------------------------#
        hsv_tuples  = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))

        self.generate()

    #---------------------------------------------------#
    #   載入模型
    #---------------------------------------------------#
    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'
        
        self.yolo_model = yolo_body([None, None, 3], self.anchors_mask, self.num_classes)
        self.yolo_model.load_weights(self.model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))
        #---------------------------------------------------------#
        #   在DecodeBox函數中，我們會對預測結果進行後處理
        #   後處理的內容包括，解碼、非極大抑制、門限篩選等
        #---------------------------------------------------------#
        self.input_image_shape = Input([2,],batch_size=1)
        inputs  = [*self.yolo_model.output, self.input_image_shape]
        outputs = Lambda(
            DecodeBox, 
            output_shape = (1,), 
            name = 'yolo_eval',
            arguments = {
                'anchors'           : self.anchors, 
                'num_classes'       : self.num_classes, 
                'input_shape'       : self.input_shape, 
                'anchor_mask'       : self.anchors_mask,
                'confidence'        : self.confidence, 
                'nms_iou'           : self.nms_iou, 
                'max_boxes'         : self.max_boxes, 
                'letterbox_image'   : self.letterbox_image
             }
        )(inputs)
        self.yolo_model = Model([self.yolo_model.input, self.input_image_shape], outputs)

    @tf.function
    def get_pred(self, image_data, input_image_shape):
        out_boxes, out_scores, out_classes = self.yolo_model([image_data, input_image_shape], training=False)
        return out_boxes, out_scores, out_classes
    #---------------------------------------------------#
    #   檢測圖片
    #---------------------------------------------------#
    def detect_image(self, image, crop = False):
        #---------------------------------------------------------#
        #   在這裡將圖像轉換成RGB圖像，防止灰度圖在預測時報錯。
        #   代碼僅僅支持RGB圖像的預測，所有其它類型的圖像都會轉化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   給圖像增加灰條，實現不失真的resize
        #   也可以直接resize進行識別
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size維度，並進行歸一化
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        #---------------------------------------------------------#
        #   將圖像輸入網絡當中進行預測！
        #---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))
        #---------------------------------------------------------#
        #   設置字體與邊框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))

        #---------------------------------------------------------#
        #   是否進行目標的裁剪
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(out_boxes)):
                top, left, bottom, right = out_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   圖像繪製
        #---------------------------------------------------------#
        for i, c in list(enumerate(out_classes)):
            predicted_class = self.class_names[int(c)]
            box             = out_boxes[i]
            score           = out_scores[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        #---------------------------------------------------------#
        #   在這裡將圖像轉換成RGB圖像，防止灰度圖在預測時報錯。
        #   代碼僅僅支持RGB圖像的預測，所有其它類型的圖像都會轉化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   給圖像增加灰條，實現不失真的resize
        #   也可以直接resize進行識別
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size維度，並進行歸一化
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)
        
        #---------------------------------------------------------#
        #   將圖像輸入網絡當中進行預測！
        #---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        t1 = time.time()
        for _ in range(test_interval):
            out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    #---------------------------------------------------#
    #   檢測圖片
    #---------------------------------------------------#
    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        #---------------------------------------------------------#
        #   在這裡將圖像轉換成RGB圖像，防止灰度圖在預測時報錯。
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   給圖像增加灰條，實現不失真的resize
        #   也可以直接resize進行識別
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size維度，並進行歸一化
        #---------------------------------------------------------#
        image_data  = np.expand_dims(preprocess_input(np.array(image_data, dtype='float32')), 0)

        #---------------------------------------------------------#
        #   將圖像輸入網絡當中進行預測！
        #---------------------------------------------------------#
        input_image_shape = np.expand_dims(np.array([image.size[1], image.size[0]], dtype='float32'), 0)
        out_boxes, out_scores, out_classes = self.get_pred(image_data, input_image_shape) 

        for i, c in enumerate(out_classes):
            predicted_class             = self.class_names[int(c)]
            try:
                score                   = str(out_scores[i].numpy())
            except:
                score                   = str(out_scores[i])
            top, left, bottom, right    = out_boxes[i]
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 
        
