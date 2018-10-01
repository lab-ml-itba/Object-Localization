import tensorflow as tf
from keras import backend as K
def custom_loss(y_true, y_pred):
    n_classes = 5
    indexes = tf.where(K.equal(y_true[:,-1], K.ones_like(y_true[:,-1])))[:,0]
    y_true_pos = tf.gather(y_true, indexes)
    y_pred_pos = tf.gather(y_pred, indexes)
    classes_cross_entropy = K.categorical_crossentropy(y_true_pos[:,:n_classes], y_pred_pos[:,:n_classes])
    bounding_box_mse = K.mean(K.square(y_pred_pos[:,n_classes:n_classes+4] - y_true_pos[:,n_classes:n_classes+4]), axis=-1)
    confidence_cross_entropy = K.mean(K.binary_crossentropy(y_true[:,n_classes+4:], y_pred[:,n_classes+4:]), axis=-1)
    return K.mean(classes_cross_entropy) + K.mean(bounding_box_mse) + K.mean(confidence_cross_entropy)

def cat_cross_entropy_loss(y_true, y_pred):
    n_classes = 5
    indexes = tf.where(K.equal(y_true[:,-1], K.ones_like(y_true[:,-1])))[:,0]
    y_true_pos = tf.gather(y_true, indexes)
    y_pred_pos = tf.gather(y_pred, indexes)
    return K.categorical_crossentropy(y_true_pos[:,:n_classes], y_pred_pos[:,:n_classes])

def bin_cross_entropy_loss(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true[:,n_classes+4:], y_pred[:,n_classes+4:]), axis=-1)
    
def boundind_box_mse_loss(y_true, y_pred):
    n_classes = 5
    indexes = tf.where(K.equal(y_true[:,-1], K.ones_like(y_true[:,-1])))[:,0]
    y_true_pos = tf.gather(y_true, indexes)
    y_pred_pos = tf.gather(y_pred, indexes)
    return K.mean(K.square(y_pred_pos[:,n_classes:n_classes+4] - y_true_pos[:,n_classes:n_classes+4]), axis=-1)

def custom_loss_IOU(y_true, y_pred):
    n_classes = 5
    indexes = tf.where(K.equal(y_true[:,-1], K.ones_like(y_true[:,-1])))[:,0]
    y_true_pos = tf.gather(y_true, indexes)
    y_pred_pos = tf.gather(y_pred, indexes)
    classes_cross_entropy = K.categorical_crossentropy(y_true_pos[:,:n_classes], y_pred_pos[:,:n_classes])
    bounding_box_IOU = IOU_loss_V2(y_true_pos, y_pred_pos)
    confidence_cross_entropy = K.mean(K.binary_crossentropy(y_true[:,n_classes+4:], y_pred[:,n_classes+4:]), axis=-1)
    return K.mean(classes_cross_entropy) + K.mean(bounding_box_IOU) + K.mean(confidence_cross_entropy)

def classes_acc(y_true, y_pred):
    n_classes = 5
    indexes = tf.where(K.equal(y_true[:,-1], K.ones_like(y_true[:,-1])))[:,0]
    y_true = tf.gather(y_true, indexes)
    y_pred = tf.gather(y_pred, indexes)
    return K.cast(K.equal(K.argmax(y_true[:,:n_classes], axis=-1),
                          K.argmax(y_pred[:,:n_classes], axis=-1)),
                  K.floatx())

def bounding_box_mse(y_true, y_pred):
    n_classes = 5
    indexes = tf.where(K.equal(y_true[:,-1], K.ones_like(y_true[:,-1])))[:,0]
    y_true = tf.gather(y_true, indexes)
    y_pred = tf.gather(y_pred, indexes)
    return K.mean(K.square(y_pred[:,n_classes:n_classes+4] - y_true[:,n_classes:n_classes+4]), axis=-1)

def confidence_acc(y_true, y_pred):
    n_classes = 5
    return K.mean(K.equal(y_true[:,n_classes+4:], K.round(y_pred[:,n_classes+4:])), axis=-1)

def iou_v2(y_true, y_pred):
    n_classes = 5
    indexes = tf.where(K.equal(y_true[:,-1], K.ones_like(y_true[:,-1])))[:,0]
    y_true = tf.gather(y_true, indexes)
    y_pred = tf.gather(y_pred, indexes)
    boxA = y_true[:,n_classes:n_classes+4]
    boxB = y_pred[:,n_classes:n_classes+4]
    xA = K.stack([boxA[:,0]-boxA[:,2]/2, boxB[:,0]-boxB[:,2]/2], axis=-1)
    yA = K.stack([boxA[:,1]-boxA[:,3]/2, boxB[:,1]-boxB[:,3]/2], axis=-1)
    xB = K.stack([boxA[:,0]+boxA[:,2]/2, boxB[:,0]+boxB[:,2]/2], axis=-1)
    yB = K.stack([boxA[:,1]+boxA[:,3]/2, boxB[:,1]+boxB[:,3]/2], axis=-1)

    xA = K.max(xA, axis=-1)
    yA = K.max(yA, axis=-1)
    xB = K.min(xB, axis=-1)
    yB = K.min(yB, axis=-1)

    interX = K.zeros_like(xB)
    interY = K.zeros_like(yB)

    interX = K.stack([interX, xB-xA], axis=-1)
    interY = K.stack([interY, yB-yA], axis=-1)

    #because of these "max", interArea may be constant 0, without gradients, and you may have problems with no gradients. 
    interX = K.max(interX, axis=-1)
    interY = K.max(interY, axis=-1)
    interArea = interX * interY

    boxAArea = (boxA[:,2]) * (boxA[:,3])    
    boxBArea = (boxB[:,2]) * (boxB[:,3]) 
    iou = interArea / (boxAArea + boxBArea - interArea)
    return iou

def IOU_loss_V2(boxA,boxB):
    iou_ = iou_v2(boxA,boxB)
    return K.ones_like(iou_)-iou_