import numpy as np
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten,Dense,Dropout,Reshape
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K 
import matplotlib.pyplot as plt
import matplotlib.patches as patches




def yolo_v1(input_shape, num_classes, S, B, l2_reg=5e-4):
    """
    input_shape: shape of Image
    num_classes: number of classes you have
    S: grid size of the output
    B: number of output boxes per grid
    l2_reg: L2 l2_reg parameter

    """
    base_model=VGG19(include_top=False,input_shape=input_shape)

    flat = Flatten()(base_model.output)
    x = Dense(512, activation='leaky_relu', kernel_regularizer=l2(l2_reg))(flat)
    x = Dropout(0.2)(x)
    x = Dense(1024, activation='leaky_relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.2)(x)
    x = Dense(2048, activation='leaky_relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dense(S * S * (B * (5 + num_classes)),activation='sigmoid',kernel_regularizer=l2(l2_reg))(x)
    output=Reshape((S,S,B *(5 + num_classes)))(x)


    model = Model(inputs=base_model.input, outputs=output)
    model.summary()
    return model



class vis_process(Callback):
    def __init__(self, data,S,B):
        super(vis_process, self).__init__()
        self.data = data
        self.S=S
        self.B=B

    def on_epoch_end(self, epoch, logs=None):
        if (epoch % 10 == 0) & (epoch >= 10):  # Run every 10 epochs
            # Select a random sample from the validation set
            image_batch, label_batch = next(iter(self.data))
            idx = np.random.randint(image_batch.shape[0])
            image, label = image_batch[idx], label_batch[idx]

            y_pred = self.model.predict(np.expand_dims(image, axis=0), verbose=0)
            y_pred=np.squeeze(y_pred,axis=0)
            
            pred_class=y_pred[...,:self.B]
            pred_box=y_pred[...,self.B:(self.B+(self.B*4))]
            pred_conf=y_pred[...,(self.B+(self.B*4)):]



            pred_box=K.reshape(pred_box*448, [self.S, self.S,self.B, 4])

            grid_size = label.shape[0]
            fig, ax = plt.subplots(1,figsize=(7,4))
            ax.imshow(image)
            ax.axis('off')  
            
            for i in range(grid_size):
                for j in range(grid_size):
                    for z in range(self.B):

                        #bbox
                        xcenter, ycenter, w, h =pred_box[i,j,z,:]
                        xmin = xcenter - (w / 2)
                        ymin = ycenter - (h / 2)

                        rect = patches.Rectangle((xmin, ymin), w, h, linewidth=3, edgecolor='r', facecolor='none')
                        ax.add_patch(rect)

                        #Confidance
                        ax.text(xmin, ymin + 10, f"Conf: {round(pred_conf[i, j,z],2)}", color='black', weight='bold')

            plt.show()