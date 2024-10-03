import warnings
warnings.filterwarnings('ignore')
import os
import sys
import json, csv
import datetime
import numpy as np
import pandas as pd
import skimage.draw
import cv2
import matplotlib.pyplot as plt
from mrcnn.visualize import display_instances
from mrcnn.config import Config
from mrcnn import model as modellib, utils
from mrcnn import updatedUtils as updatedUtils

############################################################
#  Configurations
############################################################


class CustomConfig(Config):
    """Configuration for training on the Coco  dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "object"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU i.e. IMAGES_PER_GPU = 1
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    
    IMAGE_MIN_DIM = 256

    # Number of classes - single image supported only now (including background)
    # NUM_CLASSES = 1 + len(CLASS_NAME.keys()) i.e.  Background + object(s) of interest

    NUM_CLASSES = 1 + 1  # Background + palmtree
    
    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 20

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9
    
    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 5


############################################################
#  Dataset
############################################################

class CustomDataset(utils.Dataset):

    def load_custom(self, dataset_dir, subset,objectList):
        """Load a subset of the dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        
        # Obtain the Train or validation dataset
        assert subset in ["train", "val","test"]
        dataset_dir = os.path.join(dataset_dir, subset)

        """ Load annotations
        VGG Image Annotator saves each image in the form:
        # { 'filename': '28503151_5b5b7ec140_b.jpg',
        #   'regions': {
        #       '0': {
        #           'region_attributes': {},
        #           'shape_attributes': {
        #               'all_points_x': [...],
        #               'all_points_y': [...],
        #               'name': 'polygon'}},
        #       ... more regions ...
        #   },
        #   'size': 100202
        # }
        """
        
        #Add classes
        for i in range(1,len(objectList)+1):
            self.add_class("object", i, objectList[i-1])
        
        annotations_dict = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations_dict.values())  # don't need the dict keys
        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]
        
        # Add images
        for a in annotations:
            """ Get the x, y coordinaets of points of the polygons that make up the outline of each object instance. 
            These are stores in the shape_attributes (see json format above) """
            if type(a['regions']) is dict:
                polygons = [r['shape_attributes'] for r in a['regions'].values()]
            else:
                polygons = [r['shape_attributes'] for r in a['regions']]

            """load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read the image. 
            # This is only managable since the dataset is tiny. Else you could
            # also add the image sizes in the annotation JSONS seperately after VIA labeling."""
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "object",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons
            )
            

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a class1 dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "object":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape [height, width, instance_count]
        info = self.image_info[image_id]

        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)

        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            if p['name'] == 'circle':
                rr, cc = skimage.draw.circle( p['cx'], p['cy'], p['r'])
                mask[cc, rr,i] = 1
                
            elif p['name'] == 'ellipse':
                rr, cc = skimage.draw.ellipse(p['cy'], p['cx'], p['ry'], p['rx'])
                mask[rr, cc, i] = 1
            
            elif p['name'] == 'rect':
                all_points_x = [p['x'], p['x'] + p['width'], p['x'] + p['width'], p['x']]
                all_points_y = [p['y'], p['y'] + p['height'], p['y'], p['y'] + p['height']]
                rr, cc = skimage.draw.polygon(all_points_y, all_points_x)
                mask[rr, cc, i] = 1
            
            elif (p['name'] == 'polygon') or (p['name'] == 'polyline'):
                rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
                mask[rr, cc, i] = 1
                
            else:
                raise Exception('Unknown annotation type. Supported annotation types: Rectangle, Polygon, Polyline, Ellipse.')

        # Return mask, and array of class IDs of each instance.
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
    
   
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "objects":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)

def trainModel(config,model,objectList, epochs):
    """Train the model."""
    cwd_data = os.path.join(os. getcwd()+"/dataset")
    
    # Training dataset.
    dataset_train = CustomDataset()
    dataset_train.load_custom(cwd_data, "train",objectList)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = CustomDataset()
    dataset_val.load_custom(cwd_data, "val",objectList)
    dataset_val.prepare()

    """ *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. 
    # Also, no need to train all layers, just the heads should do it."""
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=epochs,
                layers='heads')



def modelEvaluate(config_pred,model_eval,ROOT_DIR,imageSubFolder,objectList):
    imagePath = os.path.join(ROOT_DIR, "dataset")
    imageList = os.listdir(imagePath)
    dataset = CustomDataset()
    dataset.load_custom(imagePath, imageSubFolder,objectList)
    dataset.prepare()

    imageList = os.listdir(os.path.join(imagePath, imageSubFolder))
    n = len(imageList)-1
    total_gt = np.array([]) 
    total_pred = np.array([]) 
    mAP_ = [] #mAP list
    avgF1 = [] #F1 list

    class_names = ["BG"]
    for i in range(len(objectList)):
        class_names.append(objectList[i])

    """ Compute total ground truth boxes(total_gt) and total predicted boxes(total_pred) and mean average precision for each Image 
    in the test dataset"""
    
    save_dir = os.path.join(ROOT_DIR, "output")
    file = os.path.join(save_dir, "gt_pred.csv")

    if imageSubFolder == "train":
        s = "Training Set"
        with open(file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([s])
        f.close()           
    elif imageSubFolder == "val":
        s = "Validation Set"
        with open(file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([s])
        f.close()
    print(s)

    i=0
    for image_id in dataset.image_ids:
        i = i+1
        
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config_pred, image_id) #, use_mini_mask=False)
        info = dataset.image_info[image_id]

        if info["id"] in imageList:
            imageList.remove(info["id"])

        # Run the model
        results = model_eval.detect([image], verbose=0)
        r = results[0]

        #compute gt_tot and pred_tot
        gt, pred = updatedUtils.gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
        total_gt = np.append(total_gt, gt)
        total_pred = np.append(total_pred, pred)

        count = len(r['rois']) 
            
        #precision_, recall_, AP_ 
        AP_, precision_, recall_, overlap_ = updatedUtils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                            r['rois'], r['class_ids'], r['scores'], r['masks'])
        meanPrecision = np.array(precision_).mean()
        meanRecall = np.array(recall_).mean()
        F1_ =(2* (meanPrecision * meanRecall))/(meanPrecision + meanRecall)
        mAP_.append(AP_)
        avgF1.append(F1_)
    
    for j in range(len(imageList)):
        if imageList[j].split(".")[1]=="png" or imageList[j].split(".")[1]=="jpg":
            image = skimage.io.imread(os.path.join(os.path.join(imagePath, imageSubFolder), imageList[j]))
            # prediction
            results = model_eval.detect([image], verbose=0)

            # Visualize results
            r = results[0]
            count = len(r['rois'])
            if count>0:
                #precision_, recall_, AP_ 
                AP_, precision_, recall_, overlap_ = 0,0,0,0
                meanPrecision = np.array(precision_).mean()
                meanRecall = np.array(recall_).mean()
                F1_ = 0
                mAP_.append(AP_)
                avgF1.append(F1_)
                i=i+1

    total_gt=total_gt.astype(int)
    total_pred=total_pred.astype(int)

    mAP = round(sum(mAP_)/len(mAP_),3)
    F1 = round(np.array(avgF1).mean(),3)
    print("Mean Average Precision for the whole set of images = ", mAP)
    print("F1 Score for the whole set of images = ", F1)

    with open(file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Total no:of images =", n])
        writer.writerow(["Mean Average Precision for the whole set of images =", mAP])
        writer.writerow(["F1 Score for the whole set of images = ", F1])
        writer.writerow([])
    f.close()

def modelPredict(config_pred,model_eval,ROOT_DIR,imageSubFolder,objectList):
    imagePath = os.path.join(ROOT_DIR, "dataset")
    imageList = os.listdir(imagePath)
    dataset = CustomDataset()
    dataset.load_custom(imagePath, imageSubFolder,objectList)
    dataset.prepare()

    imageList = os.listdir(os.path.join(imagePath, imageSubFolder))
    n = len(imageList)-1
    total_gt = np.array([]) 
    total_pred = np.array([]) 
    mAP_ = [] #mAP list
    avgF1 = [] #F1 list

    class_names = ["BG"]
    for i in range(len(objectList)):
        class_names.append(objectList[i])

    """ Compute total ground truth boxes(total_gt) and total predicted boxes(total_pred) and mean average precision for each Image 
    in the test dataset"""
    
    print("PREDICTIONS : \n")
    print("Images: {}\nClasses: {}".format(n, dataset.class_names))
    print()

    i=0
    data = []
    for image_id in dataset.image_ids:
        d = []
        i = i+1
        
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, config_pred, image_id) #, use_mini_mask=False)
        info = dataset.image_info[image_id]
        print(i,")",info["id"])
        print()
        d.append(info["id"])

        if info["id"] in imageList:
            imageList.remove(info["id"])

        # Run the model
        results = model_eval.detect([image], verbose=0)
        r = results[0]

        #compute gt_tot and pred_tot
        gt, pred = updatedUtils.gt_pred_lists(gt_class_id, gt_bbox, r['class_ids'], r['rois'])
        total_gt = np.append(total_gt, gt)
        total_pred = np.append(total_pred, pred)

        count = len(r['rois'])
        print(count, " out of ", len(gt_bbox), " objects detected  ")
        d.append(count)
        d.append(len(gt_bbox))
        print()

        
        display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                        class_names, r['scores'] ,title=info["id"],figsize=(8, 8))     
            
        #precision_, recall_, AP_ 
        AP_, precision_, recall_, overlap_ = updatedUtils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                                            r['rois'], r['class_ids'], r['scores'], r['masks'])
        meanPrecision = np.array(precision_).mean()
        meanRecall = np.array(recall_).mean()
        F1_ =(2* (meanPrecision * meanRecall))/(meanPrecision + meanRecall)
        mAP_.append(AP_)
        avgF1.append(F1_)

        print()   
        print("Average Precision : ",round(AP_,3))
        print("Precison : ",round(meanPrecision,3))
        print("Recall : ",round(meanRecall,3))
        print("F1 Score : ",round(F1_,3))
        print()

        d.append(round(meanPrecision,3))
        d.append(round(meanRecall,3))
        d.append(round(AP_,3))
        d.append(round(F1_,3))

        data.append(d)
    
    for j in range(len(imageList)):
        if imageList[j].split(".")[1]=="png" or imageList[j].split(".")[1]=="jpg":
            d = []
            print(i+1,")",imageList[j])
            d.append(imageList[j])
            image = skimage.io.imread(os.path.join(imagePath+"/test", imageList[j]))
            # prediction
            results = model_eval.detect([image], verbose=0)

            # Visualize results
            r = results[0]
            count = len(r['rois'])
            print(count, " out of ", 0, " objects detected  ")
            d.append(count)
            d.append(0)
            print()
            display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                            class_names, r['scores'],title=imageList[j],figsize=(8, 8))
            print()
            if count>0:
                #precision_, recall_, AP_ 
                AP_, precision_, recall_, overlap_ = 0,0,0,0
                meanPrecision = np.array(precision_).mean()
                meanRecall = np.array(recall_).mean()
                F1_ = 0
                mAP_.append(AP_)
                avgF1.append(F1_)
                
                print("Average Precision : ",round(AP_,3))
                print("Precison : ",round(meanPrecision,3))
                print("Recall : ",round(meanRecall,3))
                print("F1 Score : ",round(F1_,3))
                print()
                i=i+1

                d.append(round(meanPrecision,3))
                d.append(round(meanRecall,3))
                d.append(round(AP_,3))
                d.append(round(F1_,3))
            else:
                d.append(0)
                d.append(0)
                d.append(0)
                d.append(0)
            data.append(d)

    total_gt=total_gt.astype(int)
    total_pred=total_pred.astype(int)
    mAP = round(sum(mAP_)/len(mAP_),3)
    F1 = round(np.array(avgF1).mean(),3)
    print("Mean Average Precision for the whole set of images = ", mAP)
    print("F1 Score for the whole set of images = ", F1)

    #save the vectors of gt and pred
    save_dir = os.path.join(ROOT_DIR, "output")

    df = pd.DataFrame(data, columns =['Image Name', 'Prediction', 'Ground Truth','Precision', 'Recall','Average Precision','F1 Score'])

    if imageSubFolder == "train":
        s = "Training Set"
    elif imageSubFolder == "val":
        s = "Validation Set"
    else:
        s = imageSubFolder.capitalize() +" Set"

    file = os.path.join(save_dir, "gt_pred.csv")
    with open(file, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([s])
        writer.writerow(["Total no:of images:", n])
        writer.writerow(["Mean Average Precision for the whole set of images =", mAP])
        writer.writerow(["F1 Score for the whole set of images = ", F1])
        writer.writerow([])
    f.close()

    df.to_csv(file,mode='a',index=False) 


    #gt_pred_tot_json = {"Total Groundtruth" : total_gt, "predicted box" : total_pred}
    #df = pd.DataFrame(gt_pred_tot_json)
    #if not os.path.exists(save_dir):
    #    os.makedirs(save_dir)
    #df.to_json(os.path.join(save_dir,"test_gt_pred.json"))
