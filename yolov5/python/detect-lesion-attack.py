import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from PIL import Image
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import numpy as np
from math import log10, sqrt 
from scipy.fftpack import dct,idct 
from matplotlib import pyplot as plt
####################################################################################################

class_label = ['melanoma','seborrheic keratosis','benign nevi'];
label_list = [];
correct_classified = 0;
class_id = 11;
# QueryMax =1;k=32; alpha = 0;        
QueryMax =10;k=32; alpha = 0.0002;        

# image_path = "./adv/lesion (8).jpg"
AllFrequency= False; LowFrequency = False;K_largest = True;


def  Klargest(x,k):
    nk = k *k *3;
    vx = x.reshape(-1);
    ss = np.argsort(np.multiply(-1, np.absolute(vx)),axis=0);#-1 for descending sort
    zeroed_x_idx = np.zeros(vx.shape);
    zeroed_x_idx [ss[0:nk]] = 1;

    return zeroed_x_idx
def checkDetectionWithGroundTruth(img_name, detected_label):######################

    print(img_name)
    print(detected_label)
    # print(label_list)

    for itemLabel in label_list:
        if (itemLabel.find(img_name) != -1):
            if (itemLabel.find(detected_label) != -1):
                # print(itemLabel)
                return True
            else:
                print('True labele is: ', itemLabel[12:])
                return False


def readGroundTruthFile():###############################################################
    global label_list;
    my_file = open("test_labels.txt", "r")
    label_list = my_file.readlines()
    return label_list



def PSNR_MSE(original, compressed): 
    mse = np.mean((original - compressed) ** 2) 
    if(mse == 0):  # MSE is zero means no noise is present in the signal . 
        psnr = 100
    else:
        max_pixel = 1.0
        psnr = 20 * log10(max_pixel / sqrt(mse)) 
    mse_psnr = 'mse: '+ str(mse)+' and PSNR: '+str(psnr);
    return mse_psnr

def SparseFool(legImg):###############################################################
    # print(legImg.shape[0])
    row_size = legImg.shape[1];column_size = legImg.shape[2];
    #print(legImg.shape)
    yd1 = dct(legImg, axis=0, norm="ortho");#print(yd1.shape)
    yd2 = dct(yd1, axis=1, norm="ortho");#print(yd2.shape)
    
    zeroed_x_idx  = Klargest(yd2,k)  
    # y = np.random.uniform(0,1,legImg.shape);
    y = np.random.normal(0,1,legImg.shape);
    yd1 = dct(y, axis=0, norm="ortho");
    yd2 = dct(yd1, axis=1, norm="ortho");#print(yd2.shape)
    if  K_largest: 
#        print('Sparse')
        yd3 = np.zeros(yd2.shape);
        temp = yd2.reshape(-1);
        yd3 = (np.multiply(temp,zeroed_x_idx)).reshape(3,row_size,column_size)    
        iyd1 = idct(yd3, axis=1, norm="ortho");#print(iyd1.shape)  
        iyd2 = idct(iyd1, axis=0, norm="ortho");#print(iyd2.shape)           
        
        yLF = np.sqrt(alpha) * iyd2 / np.sqrt(np.square(iyd2).mean(axis=None))#.astype(np.uint32);
        # print('mean:    ',np.mean((yLF) ** 2) )
        #print(yLF.shape)
        xAdvKlargest = np.add(yLF,legImg) 
        
        min_val = np.min(xAdvKlargest);
        if min_val<0: 
            # print(min_val);print(np.max(xAdvKlargest))
            xAdvKlargest = xAdvKlargest - min_val;
            # print('min reset!!!!')

        max_val = np.max(xAdvKlargest);
        if max_val > 1.0 and max_val !=0:
            xAdvKlargest = xAdvKlargest *(1/max_val)
            # print('max reset!!!!')
        
        str_mse_psnr = PSNR_MSE(legImg, xAdvKlargest);
        # print(str_mse_psnr)    
        



        return xAdvKlargest , str_mse_psnr


####################################################################################################



def detect(save_img=False):
    global correct_classified;
    # readGroundTruthFile();
    source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    
###########################################################################    
    global QueryMax;
    img_counter = 0;
    for path, img, im0s, vid_cap in dataset: 
        img = torch.from_numpy(img).to(device);
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0        
        img_org =img
        img_counter =img_counter+1;
        success_attack = False
        for QueryNumber in range (0,QueryMax):
            # print(img_counter,'   query: ', QueryNumber,'  out of  ',QueryMax)


    
            # print("****************************************************************************************************************")
    
            # temp, str_mse_psnr = SparseFool(img_org.numpy());#####################################################
            # img = torch.tensor(temp).float();
            imgtemp = img;
            # print("*********************************************************************************************************************************************")
    
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
    
            pred = model(img, augment=opt.augment)[0]   
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            
            # print(len(pred))
            if len(pred)>0:
                # print(pred[0].numpy()[0][5])
                temp = pred[0].numpy();
                if len(temp>0):
                    
                    if int(pred[0].numpy()[0][5]) != 2:
                        success_attack = True;
                        print('QueryNumber:  ',QueryNumber, 'attack status: ', success_attack)
                        # print('QueryNumber:  ',QueryNumber, 'attack status: ', success_attack,'   ',str_mse_psnr)
                        print(path);
                        # 
                
                        print("******************************************************")

                        im0s = (imgtemp.numpy().transpose(1,2,0)*255)
                        im0s = np.ascontiguousarray(im0s)
                        
                        # Process detections
                        for i, det in enumerate(pred):  # detections per image
                            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0);
                
                            p = Path(p);
                            save_path = str(save_dir / p.name)  # img.jpg
                            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                            s += '%gx%g ' % img.shape[2:]  # print string
                            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                            
                            if len(det):
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                                # Print results
                                for c in det[:, -1].unique():
                                    n = (det[:, -1] == c).sum()  # detections per class
                                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                                # Write results
                                for *xyxy, conf, cls in reversed(det):
                                    if save_txt:  # Write to file
                                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                        with open(txt_path + '.txt', 'a') as f:
                                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                                            
                                    if save_img or view_img:  # Add bbox to image
                                        label = f'{names[int(cls)]} {conf:.2f}'
                                        plot_one_box(xyxy, im0, label=label, color=[255, 0, 0], line_thickness=3)
                
                            if view_img:
                                cv2.imshow(str(p), im0)
                                cv2.waitKey(1)  # 1 millisecond
                
                            # Save results (image with detections)
                            if save_img:
                                if dataset.mode == 'image':
                                    cv2.imwrite(save_path,cv2.cvtColor(im0, cv2.COLOR_RGB2BGR))
                                    # print(save_path)
                                    True;
                        break;                  

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='D:/Research_in_USF/Adversarial_attack/OD_attack/yolov5-master/hadi/lesion/trained_model/weights/last.pt', help='model.pt path(s)')
    # parser.add_argument('--source', type=str, default='./hadi/lesion/test-img-one', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='./hadi/lesion/img-415-classes/2', help='source') 
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_false', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_false', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='D:/Research_in_USF/Adversarial_attack/OD_attack/yolov5-master/hadi/lesion/result', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print(opt)
    check_requirements()

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()