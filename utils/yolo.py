from tensorflow.keras.layers import (BatchNormalization, Concatenate, Input,
                                     Lambda, LeakyReLU, MaxPooling2D,
                                     UpSampling2D, ZeroPadding2D)
from tensorflow.keras.models import Model

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
