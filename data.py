import os
import numpy as np
import cv2
import albumentations as A
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
import xml.etree.ElementTree as xml 
from tensorflow.keras.utils import Sequence




def viz_dataset(image, label,grid_size=4,ax=None):
    """
    Plot images and its label which contain Bounding box in format (xmin, ymin, xmax, ymax).

    Parameters:
    - image: NumPy array representing the image.
    - label: Label matrix containing bounding box information.
    - grid_size: Size of the grid.
    -ax : axis to plot.
    -color : color of text & bbox.

    Returns:
    - None (displays the plot).
    """

    if ax==None:
        fig, ax = plt.subplots(1)
    ax.imshow(image)
    ax.axis('off')


    for i in range(grid_size):
        for j in range(grid_size):
            if label[i, j, 5] == 1:
                bbox=label[i, j, 1:5]
                xcenter, ycenter, w, h = [element*image.shape[0] for element in bbox]
                xmin = int(xcenter - w / 2)
                ymin = int(ycenter - h / 2)
                
                rect=patches.Rectangle((xmin, ymin),w,h,
                                         linewidth=2, edgecolor='b', facecolor='none')

                # Add the patch to the Axes
                ax.add_patch(rect)


class DataGenerator(Sequence):

    def __init__(self,image_set,batch_size=32, image_shape=(224,224),Transform=None,grid_size=2):
        super().__init__()

        """
        Attrs:

        image_set-> list which contain the images&annotation filenames.
        class_dictionary--->class_dictionary for classes
        Batch_size --> batch_size of Data that the generator will return back || Deafult=32
        image_shape -> Dimention of image that will get into model || Deafult=(224,224)
        preprocessing_func ---> apply preprocessing into image || Deafult=None
        grid_size ---> grid in the output of network will be grid_size*grid_size
        Methods:

        len() --> will return number of batches.
        class_index() --> will return dictonary about class maped.

        """

        self.image_set = image_set
        self.batch_size = batch_size
       

        self.image_shape = image_shape
        self.grid_size=grid_size


        self.transform=Transform


        self.images_dir = os.path.join('/kaggle','input','car-plate-detection','images')
        self.annotation=os.path.join('/kaggle','input','car-plate-detection','annotations')

        self.shuffle = True

        self.indexes = np.arange(len(self.image_set))
        self.on_epoch_end()


    def __len__(self):
        return int(np.ceil(len(self.image_set) / self.batch_size))




    def load_data(self, file_name):

        annotation_file=file_name.split('.')[0]

        # Image:
        image = cv2.imread(os.path.join(self.images_dir,file_name))
        image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        


        bboxes=[]
        file=xml.parse(os.path.join(self.annotation,annotation_file+'.xml'))
        root=file.getroot()
        objects=root.findall('object')
        for obj in objects:
          bbox=obj.find('bndbox')
          xmin=int(bbox.find('xmin').text)
          ymin=int(bbox.find('ymin').text)
          xmax=int(bbox.find('xmax').text)
          ymax=int(bbox.find('ymax').text)
          xcenter=(xmin+xmax)/2/image.shape[1]
          ycenter=(ymin+ymax)/2/image.shape[0]
          width=(xmax-xmin)/image.shape[1]
          hieght=(ymax-ymin)/image.shape[0]
          bboxes.append([xcenter,ycenter,width,hieght])


        if self.transform is None:
          self.transform  = A.Compose([A.Resize(self.image_shape[0],self.image_shape[1])],bbox_params=A.BboxParams(format='yolo',label_fields=[]))


        transformed = self.transform(image=image, bboxes=bboxes)
        transformed_image = transformed['image']
        transformed_bboxes = transformed['bboxes']

        


        # Label:
        label_matrix=np.zeros((self.grid_size,self.grid_size,6))

        for box in transformed_bboxes:


          xcenter,ycenter,width,hieght=box



          loc = [int(self.grid_size * xcenter), int(self.grid_size * ycenter)]


          if label_matrix[loc[1], loc[0], 5] == 0:
             label_matrix[loc[1], loc[0], 0] = 1
             label_matrix[loc[1], loc[0], 1:5] = box
             label_matrix[loc[1], loc[0], 5] = 1

        return transformed_image,label_matrix


    def __getitem__(self, index):
            start = index * self.batch_size
            end = (index + 1) * self.batch_size
            indices = np.arange(start, min(end, len(self.image_set)))

            images = np.zeros((self.batch_size, self.image_shape[0], self.image_shape[0], 3), dtype=np.float32)
            labels = np.zeros((self.batch_size, self.grid_size, self.grid_size,6), dtype=np.float32)

            for i, idx  in enumerate(indices):
                image, label = self.load_data(self.image_set[idx])
                images[i] = image/255.
                labels[i] = label

            return images, labels


    def on_epoch_end(self):
        # Shuffle indexes at the end of each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)



    def on_epoch_end(self):
        # Shuffle indexes at the end of each epoch
        if self.shuffle:
            np.random.shuffle(self.indexes)