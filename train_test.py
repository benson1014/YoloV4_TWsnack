from tensorflow.keras.layers import (BatchNormalization, Concatenate, Input,
                                     Lambda, LeakyReLU, MaxPooling2D,
                                     UpSampling2D, ZeroPadding2D)
from tensorflow.keras.models import Model

import os
os.environ['CUDA_VISIBLE_DEVICES']="0,1,2,3,4,5,6,7"
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

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


###########################################################################################################
# from nets.yolo import get_train_model, yolo_body
#####  定義函式  #####
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




###############################################################################
#  主程式
###############################################################################

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

def get_train_model(model_body, input_shape, num_classes, anchors, anchors_mask, label_smoothing):
    y_true = [Input(shape = (input_shape[0] // {0:32, 1:16, 2:8}[l], input_shape[1] // {0:32, 1:16, 2:8}[l], \
                                len(anchors_mask[l]), num_classes + 5)) for l in range(len(anchors_mask))]
    model_loss  = Lambda(
        yolo_loss, 
        output_shape    = (1, ), 
        name            = 'yolo_loss', 
        arguments       = {
            'input_shape'       : input_shape, 
            'anchors'           : anchors, 
            'anchors_mask'      : anchors_mask, 
            'num_classes'       : num_classes, 
            'balance'           : [0.4, 1.0, 4],
            'box_ratio'         : 0.05,
            'obj_ratio'         : 5 * (input_shape[0] * input_shape[1]) / (416 ** 2), 
            'cls_ratio'         : 1 * (num_classes / 80),
            'label_smoothing'   : label_smoothing
        }
    )([*model_body.output, *y_true])
    model       = Model([model_body.input, *y_true], model_loss)
    return model


###########################################################################################################
# from nets.yolo_training import get_lr_scheduler

def get_lr_scheduler(lr_decay_type, lr, min_lr, total_iters, warmup_iters_ratio = 0.1, warmup_lr_ratio = 0.1, no_aug_iter_ratio = 0.3, step_num = 10):
    def yolox_warm_cos_lr(lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter, iters):
        if iters <= warmup_total_iters:
            # lr = (lr - warmup_lr_start) * iters / float(warmup_total_iters) + warmup_lr_start
            lr = (lr - warmup_lr_start) * pow(iters / float(warmup_total_iters), 2
            ) + warmup_lr_start
        elif iters >= total_iters - no_aug_iter:
            lr = min_lr
        else:
            lr = min_lr + 0.5 * (lr - min_lr) * (
                1.0
                + math.cos(
                    math.pi
                    * (iters - warmup_total_iters)
                    / (total_iters - warmup_total_iters - no_aug_iter)
                )
            )
        return lr

    def step_lr(lr, decay_rate, step_size, iters):
        if step_size < 1:
            raise ValueError("step_size must above 1.")
        n       = iters // step_size
        out_lr  = lr * decay_rate ** n
        return out_lr

    if lr_decay_type == "cos":
        warmup_total_iters  = min(max(warmup_iters_ratio * total_iters, 1), 3)
        warmup_lr_start     = max(warmup_lr_ratio * lr, 1e-6)
        no_aug_iter         = min(max(no_aug_iter_ratio * total_iters, 1), 15)
        func = partial(yolox_warm_cos_lr ,lr, min_lr, total_iters, warmup_total_iters, warmup_lr_start, no_aug_iter)
    else:
        decay_rate  = (min_lr / lr) ** (1 / (step_num - 1))
        step_size   = total_iters / step_num
        func = partial(step_lr, lr, decay_rate, step_size)

    return func


###########################################################################################################
# from utils.callbacks import LossHistory, ModelCheckpoint
import math
import os
import warnings

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import scipy.signal
from tensorflow import keras
from tensorflow.keras import backend as K


class LossHistory(keras.callbacks.Callback):
    def __init__(self, log_dir):
        self.log_dir    = log_dir
        self.losses     = []
        self.val_loss   = []
        
        os.makedirs(self.log_dir)

    def on_epoch_end(self, epoch, logs={}):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(logs.get('loss')))
            f.write("\n")
        with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
            f.write(str(logs.get('val_loss')))
            f.write("\n")
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('A Loss Curve')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class ExponentDecayScheduler(keras.callbacks.Callback):
    def __init__(self,
                 decay_rate,
                 verbose=0):
        super(ExponentDecayScheduler, self).__init__()
        self.decay_rate         = decay_rate
        self.verbose            = verbose
        self.learning_rates     = []

    def on_epoch_end(self, batch, logs=None):
        learning_rate = K.get_value(self.model.optimizer.lr) * self.decay_rate
        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('Setting learning rate to %s.' % (learning_rate))

class WarmUpCosineDecayScheduler(keras.callbacks.Callback):
    def __init__(self, T_max, eta_min=0, verbose=0):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.T_max      = T_max
        self.eta_min    = eta_min
        self.verbose    = verbose
        self.init_lr    = 0
        self.last_epoch = 0

    def on_train_begin(self, batch, logs=None):
        self.init_lr = K.get_value(self.model.optimizer.lr)

    def on_epoch_end(self, batch, logs=None):
        learning_rate = self.eta_min + (self.init_lr - self.eta_min) * (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
        self.last_epoch += 1

        K.set_value(self.model.optimizer.lr, learning_rate)
        if self.verbose > 0:
            print('Setting learning rate to %s.' % (learning_rate))
    
class ModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(ModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


###########################################################################################################
# from utils.dataloader import YoloDatasets
import math
from random import sample, shuffle

import cv2
import numpy as np
from PIL import Image
from tensorflow import keras

# from utils.utils import cvtColor, preprocess_input
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


def preprocess_input(image):
    image /= 255.0
    return image
# from utils.utils import cvtColor, preprocess_input


class YoloDatasets(keras.utils.Sequence):
    def __init__(self, annotation_lines, input_shape, anchors, batch_size, num_classes, anchors_mask, epoch_now, epoch_length, mosaic, train, mosaic_ratio = 0.7):
        self.annotation_lines   = annotation_lines
        self.length             = len(self.annotation_lines)
        
        self.input_shape        = input_shape
        self.anchors            = anchors
        self.batch_size         = batch_size
        self.num_classes        = num_classes
        self.anchors_mask       = anchors_mask
        self.epoch_now          = epoch_now - 1
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.train              = train
        self.mosaic_ratio       = mosaic_ratio

        self.threshold          = 4

    def __len__(self):
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        image_data  = []
        box_data    = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):  
            i           = i % self.length
            #---------------------------------------------------#
            #   训练时进行数据的随机增强
            #   验证时不进行数据的随机增强
            #---------------------------------------------------#
            if self.mosaic:
                if self.rand() < 0.5 and self.epoch_now < self.epoch_length * self.mosaic_ratio:
                    lines = sample(self.annotation_lines, 3)
                    lines.append(self.annotation_lines[i])
                    shuffle(lines)
                    image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
                else:
                    image, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)
            else:
                image, box  = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)
            image_data.append(preprocess_input(np.array(image, np.float32)))
            box_data.append(box)

        image_data  = np.array(image_data)
        box_data    = np.array(box_data)
        y_true      = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
        return [image_data, *y_true], np.zeros(self.batch_size)

    def generate(self):
        i = 0
        while True:
            image_data  = []
            box_data    = []
            for b in range(self.batch_size):
                if i==0:
                    np.random.shuffle(self.annotation_lines)
                #---------------------------------------------------#
                #   训练时进行数据的随机增强
                #   验证时不进行数据的随机增强
                #---------------------------------------------------#
                if self.mosaic:
                    if self.rand() < 0.5 and self.epoch_now < self.epoch_length * self.mosaic_ratio:
                        lines = sample(self.annotation_lines, 3)
                        lines.append(self.annotation_lines[i])
                        shuffle(lines)
                        image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)
                    else:
                        image, box = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)
                else:
                    image, box  = self.get_random_data(self.annotation_lines[i], self.input_shape, random = self.train)

                i           = (i+1) % self.length
                image_data.append(preprocess_input(np.array(image, np.float32)))
                box_data.append(box)
            image_data  = np.array(image_data)
            box_data    = np.array(box_data)
            y_true      = self.preprocess_true_boxes(box_data, self.input_shape, self.anchors, self.num_classes)
            yield image_data, y_true[0], y_true[1], y_true[2]
            
    def on_epoch_end(self):
        self.epoch_now += 1
        shuffle(self.annotation_lines)

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, max_boxes=500, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            box_data = np.zeros((max_boxes,5))
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0]  = 0
                box[:, 2][box[:, 2]>w]      = w
                box[:, 3][box[:, 3]>h]      = h
                box_w   = box[:, 2] - box[:, 0]
                box_h   = box[:, 3] - box[:, 1]
                box     = box[np.logical_and(box_w>1, box_h>1)]
                if len(box)>max_boxes: box = box[:max_boxes]
                box_data[:len(box)] = box

            return image_data, box_data
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        box_data = np.zeros((max_boxes,5))
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box
            if len(box)>max_boxes: box = box[:max_boxes]
            box_data[:len(box)] = box
        
        return image_data, box_data

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, max_boxes=500, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = [] 
        box_datas   = []
        index       = 0
        for line in annotation_line:
            #---------------------------------#
            #   每一行进行分割
            #---------------------------------#
            line_content = line.split()
            #---------------------------------#
            #   打开图片
            #---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)
            
            #---------------------------------#
            #   图片的大小
            #---------------------------------#
            iw, ih = image.size
            #---------------------------------#
            #   保存框的位置
            #---------------------------------#
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
            
            #---------------------------------#
            #   是否翻转图片
            #---------------------------------#
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0,2]] = iw - box[:, [2,0]]

            #------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            #------------------------------------------#
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            #-----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            #-----------------------------------------------#
            if index == 0:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif index == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 2:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            elif index == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh
            
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            #---------------------------------#
            #   对box进行重新处理
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)

        #---------------------------------#
        #   将图片分割，放在一起
        #---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image       = np.array(new_image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype           = new_image.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对框进行进一步的处理
        #---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        #---------------------------------#
        #   将box进行调整
        #---------------------------------#
        box_data = np.zeros((max_boxes, 5))
        if len(new_boxes)>0:
            if len(new_boxes)>max_boxes: new_boxes = new_boxes[:max_boxes]
            box_data[:len(new_boxes)] = new_boxes
        return new_image, box_data

    def preprocess_true_boxes(self, true_boxes, input_shape, anchors, num_classes):
        assert (true_boxes[..., 4]<num_classes).all(), 'class id must be less than num_classes'
        #-----------------------------------------------------------#
        #   获得框的坐标和图片的大小
        #-----------------------------------------------------------#
        true_boxes  = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')
        
        #-----------------------------------------------------------#
        #   一共有三个特征层数
        #-----------------------------------------------------------#
        num_layers  = len(self.anchors_mask)
        #-----------------------------------------------------------#
        #   m为图片数量，grid_shapes为网格的shape
        #-----------------------------------------------------------#
        m           = true_boxes.shape[0]
        grid_shapes = [input_shape // {0:32, 1:16, 2:8}[l] for l in range(num_layers)]
        #-----------------------------------------------------------#
        #   y_true的格式为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
        #-----------------------------------------------------------#
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(self.anchors_mask[l]), 5 + num_classes),
                    dtype='float32') for l in range(num_layers)]

        #-----------------------------------------------------------#
        #   通过计算获得真实框的中心和宽高
        #   中心点(m,n,2) 宽高(m,n,2)
        #-----------------------------------------------------------#
        boxes_xy = (true_boxes[..., 0:2] + true_boxes[..., 2:4]) // 2
        boxes_wh =  true_boxes[..., 2:4] - true_boxes[..., 0:2]
        #-----------------------------------------------------------#
        #   将真实框归一化到小数形式
        #-----------------------------------------------------------#
        true_boxes[..., 0:2] = boxes_xy / input_shape[::-1]
        true_boxes[..., 2:4] = boxes_wh / input_shape[::-1]

        #-----------------------------------------------------------#
        #   [9,2] -> [1,9,2]
        #-----------------------------------------------------------#
        anchors         = np.expand_dims(anchors, 0)
        anchor_maxes    = anchors / 2.
        anchor_mins     = -anchor_maxes

        #-----------------------------------------------------------#
        #   长宽要大于0才有效
        #-----------------------------------------------------------#
        valid_mask = boxes_wh[..., 0]>0

        for b in range(m):
            #-----------------------------------------------------------#
            #   对每一张图进行处理
            #-----------------------------------------------------------#
            wh = boxes_wh[b, valid_mask[b]]
            if len(wh) == 0: continue
            #-----------------------------------------------------------#
            #   [n,2] -> [n,1,2]
            #-----------------------------------------------------------#
            wh          = np.expand_dims(wh, -2)
            box_maxes   = wh / 2.
            box_mins    = - box_maxes

            #-----------------------------------------------------------#
            #   计算所有真实框和先验框的交并比
            #   intersect_area  [n,9]
            #   box_area        [n,1]
            #   anchor_area     [1,9]
            #   iou             [n,9]
            #-----------------------------------------------------------#
            intersect_mins  = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area  = intersect_wh[..., 0] * intersect_wh[..., 1]

            box_area    = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]

            iou = intersect_area / (box_area + anchor_area - intersect_area)
            #-----------------------------------------------------------#
            #   维度是[n,] 感谢 消尽不死鸟 的提醒
            #-----------------------------------------------------------#
            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                #-----------------------------------------------------------#
                #   找到每个真实框所属的特征层
                #-----------------------------------------------------------#
                for l in range(num_layers):
                    if n in self.anchors_mask[l]:
                        #-----------------------------------------------------------#
                        #   floor用于向下取整，找到真实框所属的特征层对应的x、y轴坐标
                        #-----------------------------------------------------------#
                        i = np.floor(true_boxes[b,t,0] * grid_shapes[l][1]).astype('int32')
                        j = np.floor(true_boxes[b,t,1] * grid_shapes[l][0]).astype('int32')
                        #-----------------------------------------------------------#
                        #   k指的的当前这个特征点的第k个先验框
                        #-----------------------------------------------------------#
                        k = self.anchors_mask[l].index(n)
                        #-----------------------------------------------------------#
                        #   c指的是当前这个真实框的种类
                        #-----------------------------------------------------------#
                        c = true_boxes[b, t, 4].astype('int32')
                        #-----------------------------------------------------------#
                        #   y_true的shape为(m,13,13,3,85)(m,26,26,3,85)(m,52,52,3,85)
                        #   最后的85可以拆分成4+1+80，4代表的是框的中心与宽高、
                        #   1代表的是置信度、80代表的是种类
                        #-----------------------------------------------------------#
                        y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 5+c] = 1

        return y_true


###########################################################################################################
# from utils.utils import get_anchors, get_classes

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


###########################################################################################################
# from utils.utils_fit import fit_one_epoch
from tqdm import tqdm

#------------------------------#
#   防止bug
#------------------------------#
def get_train_step_fn(input_shape, anchors, anchors_mask, num_classes, label_smoothing):
    @tf.function
    def train_step(imgs, targets, net, optimizer):
        with tf.GradientTape() as tape:
            #------------------------------#
            #   计算loss
            #------------------------------#
            P5_output, P4_output, P3_output = net(imgs, training=True)
            args        = [P5_output, P4_output, P3_output] + targets
            loss_value  = yolo_loss(
                args, input_shape, anchors, anchors_mask, num_classes, 
                label_smoothing = label_smoothing,
                balance         = [0.4, 1.0, 4],
                box_ratio       = 0.05, 
                obj_ratio       = 5 * (input_shape[0] * input_shape[1]) / (416 ** 2),
                cls_ratio       = 1 * (num_classes / 80)
            )
            #------------------------------#
            #   添加上l2正则化参数
            #------------------------------#
            loss_value  = tf.reduce_sum(net.losses) + loss_value
        grads = tape.gradient(loss_value, net.trainable_variables)
        optimizer.apply_gradients(zip(grads, net.trainable_variables))
        return loss_value
    return train_step


def fit_one_epoch(net, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, 
            input_shape, anchors, anchors_mask, num_classes, label_smoothing, save_period):
    train_step  = get_train_step_fn(input_shape, anchors, anchors_mask, num_classes, label_smoothing)
    loss        = 0
    val_loss    = 0
    print('Start Train')
    with tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_step:
                break
            images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
            targets     = [target0, target1, target2]
            targets     = [tf.convert_to_tensor(target) for target in targets]
            loss_value  = train_step(images, targets, net, optimizer)
            loss        = loss + loss_value

            pbar.set_postfix(**{'total_loss': float(loss) / (iteration + 1), 
                                'lr'        : optimizer.lr.numpy()})
            pbar.update(1)
    print('Finish Train')
            
    print('Start Validation')
    with tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen_val):
            if iteration >= epoch_step_val:
                break
            images, target0, target1, target2 = batch[0], batch[1], batch[2], batch[3]
            targets     = [target0, target1, target2]
            targets     = [tf.convert_to_tensor(target) for target in targets]
            #------------------------------#
            #   计算loss
            #------------------------------#
            P5_output, P4_output, P3_output = net(images)
            args        = [P5_output, P4_output, P3_output] + targets
            loss_value  = yolo_loss(args, input_shape, anchors, anchors_mask, num_classes, label_smoothing=label_smoothing)
            #------------------------------#
            #   添加上l2正则化参数
            #------------------------------#
            loss_value  = tf.reduce_sum(net.losses) + loss_value
            val_loss = val_loss + loss_value

            pbar.set_postfix(**{'total_loss': float(val_loss) / (iteration + 1)})
            pbar.update(1)
    print('Finish Validation')

    logs = {'loss': loss.numpy() / epoch_step, 'val_loss': val_loss.numpy() / epoch_step_val}
    loss_history.on_epoch_end([], logs)
    print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
    print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))
    if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        net.save_weights('logs/ep%03d-loss%.3f-val_loss%.3f.h5' % (epoch + 1, loss / epoch_step, val_loss / epoch_step_val))

