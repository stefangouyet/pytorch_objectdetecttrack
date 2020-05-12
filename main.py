from models import *
from utils import *

import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

import pathlib

class AnimalImageDetection():
    def __init__(self,input_path,output_path):
        
        self.config_path = 'config/yolov3.cfg'
        self.weights_path = 'config/yolov3.weights'
        self.class_path = 'config/coco.names'
        self.img_size = 416
        self.conf_thres = 0.8
        self.nms_thres = 0.4
        self.Tensor = torch.FloatTensor
        self.model = Darknet(self.config_path, 
                        img_size=self.img_size)
        self.model.load_weights(self.weights_path)
        #model.cuda()

        self.model.eval()
        
        self.path= pathlib.PurePath(os.getcwd())
        self.input_path = str(self.path / input_path) + '/'
        self.output_path = str(self.path / output_path) + '/'
        
        print(self.path)
        print(input_path)
        print(self.input_path)
        
    
    def detect_image(self,img):
        # scale and pad image
        ratio = min(self.img_size/img.size[0], self.img_size/img.size[1])
        imw = round(img.size[0] * ratio)
        imh = round(img.size[1] * ratio)
        img_transforms = transforms.Compose([ transforms.Resize((imh, imw)),
                                                transforms.Pad((max(int((imh-imw)/2),0), 
                                                                max(int((imw-imh)/2),0), 
                                                                max(int((imh-imw)/2),0), 
                                                                max(int((imw-imh)/2),0)),
                                                                (128,128,128)),
                                                                 transforms.ToTensor(),
                                                                 ])
        # convert image to Tensor
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        input_img = Variable(image_tensor.type(self.Tensor))
        
        # run inference on the model and get detections
        with torch.no_grad():
            detections = self.model(input_img)
            detections = utils.non_max_suppression(detections, 80, 
                                                   self.conf_thres, 
                                                   self.nms_thres)
        return detections[0]

    

    
    def cnn_model(self,img_path):

        # Load model and weights
        

        classes = utils.load_classes(self.class_path)

        #Tensor = torch.cuda.FloatTensor
        
        
        # load image and get detections
        #img_path = "images/blueangels.jpg"
        prev_time = time.time()
        img = Image.open(img_path)
        print(img)
        detections = self.detect_image(img)

        inference_time = datetime.timedelta(seconds=time.time() - prev_time)
        print ('Inference Time: %s' % (inference_time))
        # Get bounding-box colors
        cmap = plt.get_cmap('tab20b')
        colors = [cmap(i) for i in np.linspace(0, 1, 20)]
        img = np.array(img)
        plt.figure()
        fig, ax = plt.subplots(1, figsize=(12,9))
        ax.imshow(img)
        pad_x = max(img.shape[0] - img.shape[1], 0) * (self.img_size / max(img.shape))
        pad_y = max(img.shape[1] - img.shape[0], 0) * (self.img_size / max(img.shape))
        unpad_h = self.img_size - pad_y
        unpad_w = self.img_size - pad_x
        
        if detections is not None:
            unique_labels = detections[:, -1].cpu().unique()
            n_cls_preds = len(unique_labels)
            print(n_cls_preds)
            bbox_colors = random.sample(colors, n_cls_preds)
            # browse detections and draw bounding boxes
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                box_h = ((y2 - y1) / unpad_h) * img.shape[0]
                box_w = ((x2 - x1) / unpad_w) * img.shape[1]
                y1 = ((y1 - pad_y // 2) / unpad_h) * img.shape[0]
                x1 = ((x1 - pad_x // 2) / unpad_w) * img.shape[1]
                color = bbox_colors[int(np.where(
                     unique_labels == int(cls_pred))[0])]
                bbox = patches.Rectangle((x1, y1), box_w, box_h,
                     linewidth=2, edgecolor=color, facecolor='none')
                ax.add_patch(bbox)
                plt.text(x1, y1, s=classes[int(cls_pred)], 
                        color='white', verticalalignment='top',
                        bbox={'color': color, 'pad': 0})
                print(classes[int(cls_pred)])
                

        #only output if picture of "zebra"
        if classes[int(cls_pred)] == 'zebra':
            print(classes[int(cls_pred)])
            plt.axis('off')
            # save image
            plt.savefig(img_path.replace('input','output').replace(".jpg", "-det.jpg"),        
                              bbox_inches='tight', pad_inches=0.0)
            plt.show()
            
    def run_model(self):
        print(self.input_path)
        for filename in os.listdir(self.input_path):
            if filename.endswith('.jpg'):
                print(self.input_path + filename)
                try:
                    self.cnn_model(self.input_path + filename)
                except Exception as e:
                    print("error: {}".format(e))
                    pass

        

animal_detection = AnimalImageDetection(input_path = 'images-bucket/input/', 
                                        output_path = 'images-bucket/output/')

if '__name__ = main':
    animal_detection.run_model()
    
    
