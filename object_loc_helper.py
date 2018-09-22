synset_file = 'dataset/LOC_synset_mapping.txt'

def get_class_names_map():
    f = open(synset_file,'r')
    synset_lines = f.readlines()
    f.close()
    synset_dict = {}
    for line in synset_lines:
        key = line.replace('\n','').split()[0]
        synset_dict[key] = line.replace('\n','').replace(key, '').strip()
    return synset_dict

import keras
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np

class PlotLosses(keras.callbacks.Callback):
    def __init__(self, plot_interval=1):
        self.plot_interval = plot_interval
    
    def on_train_begin(self, logs={}):
        print('Begin training')
        self.i = 0
        self.x = []
        self.total_losses = {}
        self.acc = {}
        self.abserrors = {}
        self.cat_output_loss = {}
        self.bb_loss = {}
        self.ious = {}
        self.logs = []
    
    def on_epoch_end(self, epoch, logs={}):
        print(logs)
        for k,v in logs.items():
            if k in ['loss', 'val_loss']:
                if k not in self.total_losses:
                    self.total_losses[k] = []
                self.total_losses[k].append(v)
            elif 'category_output_loss' in k:
                if k not in self.cat_output_loss:
                    self.cat_output_loss[k] = []
                self.cat_output_loss[k].append(v)
            elif 'bounding_box_loss' in k:
                if k not in self.bb_loss:
                    self.bb_loss[k] = []
                self.bb_loss[k].append(v)
            elif 'acc' in k:
                if k not in self.acc:
                    self.acc[k] = []
                self.acc[k].append(v)
            elif 'error' in k:
                if k not in self.abserrors:
                    self.abserrors[k] = []
                self.abserrors[k].append(v)
            elif 'iou' in k:
                if k not in self.ious:
                    self.ious[k] = []
                self.ious[k].append(v)
                
        self.logs.append(logs)
        self.x.append(self.i)
        self.i += 1
        if (epoch%self.plot_interval==0):
            clear_output(wait=True)
            f, ((ax1, ax2),(ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, sharex=True, figsize=(20,15))
            
            for k,v in self.total_losses.items():
                if 'val' in k:
                    ax1.plot(self.x, v, label=k, ls='-.', color='b')
                else:
                    ax1.plot(self.x, v, label=k, color='r')
            
            for k,v in self.cat_output_loss.items():
                if 'val' in k:
                    ax2.plot(self.x, v, label=k, ls='-.', color='b')
                else:
                    ax2.plot(self.x, v, label=k, color='r')
            
            
            for k,v in self.bb_loss.items():
                if 'val' in k:
                    ax3.plot(self.x, v, label=k, ls='-.', color='b')
                else:
                    ax3.plot(self.x, v, label=k, color='r')
            
            for k,v in self.abserrors.items():
                if 'val' in k:
                    ax4.plot(self.x, v, label=k, ls='-.', color='b')
                else:
                    ax4.plot(self.x, v, label=k, color='r')
            
            for k,v in self.acc.items():
                if 'val' in k:
                    ax5.plot(self.x, v, label=k, ls='-.', color='b')
                else:
                    ax5.plot(self.x, v, label=k, color='r')
            
            for k,v in self.ious.items():
                if 'val' in k:
                    ax6.plot(self.x, v, label=k, ls='-.', color='b')
                else:
                    ax6.plot(self.x, v, label=k, color='r')
                    
                    
            ax1.legend()
            ax2.legend()
            ax3.legend()
            ax4.legend()
            ax5.legend()
            ax6.legend()
            
            plt.show()
            
from PIL import Image
import matplotlib.patches as patches

def plot_sample_class(class_index, annotations_dict, synset_dict, data_folder):
    img_class = list(annotations_dict.keys())[class_index]
    print(img_class, synset_dict[img_class])
    image_filenames = list(annotations_dict[img_class].keys())
    print('Cantidad de imagenes de esta clase:', len(image_filenames))

    fig, axs = plt.subplots(4,4, figsize=(20,10))
    axs = axs.flatten()
    for i, image_file_id in enumerate(image_filenames[:16]):
        image_file = data_folder+'/train/'+img_class+'/' +image_file_id+'.JPEG'
        image = Image.open(image_file)
        axs[i].imshow(image)
        bounding_box = annotations_dict[img_class][image_file_id]['bounding_boxes'][0]
        rect = patches.Rectangle(bounding_box[:2],bounding_box[2]-bounding_box[0],bounding_box[3]-bounding_box[1],linewidth=5, edgecolor='y',facecolor='none')
        axs[i].add_patch(rect)
    plt.show()
    
## Generator ###
from  keras.utils import Sequence
from keras.preprocessing.image import ImageDataGenerator
class GeneratorMultipleOutputs(Sequence):
    def __init__(self, annotations_dict, folder, batch_size, flip = 'no_flip', get_filenames = False):
        # flip = {no_flip, always, random}
        self.flip = flip
        self.get_filenames = get_filenames
        np.random.seed(seed=40)
        self.annotations_dict = annotations_dict
        datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=False)
        self.generator = datagen.flow_from_directory(
            directory=folder,
            target_size=(375, 500),
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
    def get_image_object_center(self):
        bboxes = []
        batch_index = self.generator.batch_index
        if self.generator.batch_index == 0:
            batch_index = self.__len__()
        batch_filenames = np.array(self.generator.filenames)[self.generator.index_array][(batch_index-1)*self.generator.batch_size:batch_index*self.generator.batch_size]
        #print(self.generator.batch_index, self.generator.total_batches_seen, len(batch_filenames), batch_index)
        for filename in batch_filenames:
            arr = filename.split('/')
            class_id = arr[0]
            image_idx = arr[1].split('.')[0]
            # print(filename, class_id, image_idx)
            bboxes.append(self.annotations_dict[class_id][image_idx]['bounding_boxes'][0])
        bboxes = np.array(bboxes)
        img_width = self.annotations_dict[class_id][image_idx]['width']
        img_height = self.annotations_dict[class_id][image_idx]['height']
        box_width = bboxes[:,2]-bboxes[:,0]
        box_height = bboxes[:,3]-bboxes[:,1]
        centerX = (bboxes[:,0]+(box_width)/2)/img_width
        centerY = (bboxes[:,1]+(box_height)/2)/img_height
        return centerX, centerY, box_width/img_width, box_height/img_height, batch_filenames
    def __len__(self):
        return int(np.ceil(self.generator.samples / float(self.generator.batch_size)))
    def __getitem__(self, idx):
        data = next(self.generator)
        centerX, centerY, width, height, batch_filenames = self.get_image_object_center()
        if self.flip == 'random':
            inices_to_flip = np.random.randint(0, 2, data[0].shape[0]).nonzero()
            data[0][inices_to_flip] = np.flip(data[0][inices_to_flip], axis = 2)
            centerX[inices_to_flip] = 1 - centerX[inices_to_flip]
        elif self.flip == 'always':
            data[0][:] = np.flip(data[0][:], axis = 2)
            centerX = 1 - centerX
        if self.get_filenames:
            return (data[0], [data[1], np.array([centerX, centerY, width, height]).T], batch_filenames)
        else:    
            return (data[0], [data[1], np.array([centerX, centerY, width, height]).T])
    def __next__(self):
        return self.__getitem__(0)
    def __iter__(self):
        return self
    
### Conv Models ########

from keras.layers import Activation, Dropout, Dense, Input, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

def get_conv_layer(x, filters = 32, filter_size = (3,3), pool_size=(2,2)):
    conv = Conv2D(filters, filter_size)(x)
    BN = BatchNormalization()(conv)
    act = Activation('relu')(BN)
    out = MaxPooling2D(pool_size=pool_size)(act)
    # DO1 = Dropout(0.25)(maxPool1)
    return out

def get_simple_model_common_part(input_shape=(375, 500, 3)):
    x = Input(shape=(375, 500, 3))
    l1 = get_conv_layer(x, filters = 32, filter_size = (3,3), pool_size=(2,2))
    
    l2 = get_conv_layer(l1, filters = 64, filter_size = (3,3), pool_size=(2,2))
    
    l3 = get_conv_layer(l2, filters = 128, filter_size = (3,3), pool_size=(2,2))
    
    l4 = get_conv_layer(l3, filters = 256, filter_size = (3,3), pool_size=(2,2))
    
    l5 = get_conv_layer(l4, filters = 512, filter_size = (3,3), pool_size=(2,2))

    GAP = GlobalAveragePooling2D()(l5)
    last_layer = Dropout(0.25)(GAP)
    return last_layer, x

def get_simple_model(input_shape=(375, 500, 3), n_classes=5):
    last_layer, input_layer = get_simple_model_common_part(input_shape=input_shape)
    classification = Dense(n_classes, activation='softmax', name='category_output')(last_layer) #kernel_constraint=max_norm(2.)
    bounding_box = Dense(4, name='bounding_box')(last_layer)
    model = Model(inputs=input_layer, outputs=[classification, bounding_box])
    return model

#### Custom metrics

from keras import backend as K
def iou(boxA,boxB):
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

## IOU en numpy
def getBB_area(bb):
    IntersectionArea = (bb[:,2] - bb[:,0])*(bb[:,3] - bb[:,1])
    return IntersectionArea

def getIUO(bb1, bb2):
    intersection_bb = np.array([np.vstack([bb1[:,0], bb2[:,0]]).max(axis=0),
        np.vstack([bb1[:,1], bb2[:,1]]).max(axis=0),
        np.vstack([bb1[:,2], bb2[:,2]]).min(axis=0),
        np.vstack([bb1[:,3], bb2[:,3]]).min(axis=0)]).T
    no_intersec = 1*(intersection_bb[:,3]-intersection_bb[:,1]>0)*(intersection_bb[:,2]-intersection_bb[:,0]>0)
    intersection_bb = (intersection_bb.T * no_intersec).T
    IntersectionArea = no_intersec*getBB_area(intersection_bb)
    IOU = IntersectionArea/(getBB_area(bb1) + getBB_area(bb2) - IntersectionArea)
    return IOU, intersection_bb