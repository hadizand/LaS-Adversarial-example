"""
Created on Sun May  3 09:32:13 2020

@author: Hadi Zanddizari
email: hadiz@usf.edu
"""
##############################################################################################################
from __future__ import print_function
import tensorflow as tf
from PIL import Image
import glob
import numpy as np
import os
from matplotlib import pyplot as plt
import scipy.misc
from skimage.transform import resize
import cv2
from matplotlib import pyplot as plt
from scipy.fftpack import dct,idct 
##############################################################################################################
def load_images_from_folder(folder):
    images = [];listName = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
            listName.append(filename)
    return images,listName
##############################################################################################################
def  Klargest(x,k):
    nk = k *k *3;
    vx = x.reshape(-1);
    ss = np.argsort(np.multiply(-1, np.absolute(vx)),axis=0);#-1 for descending sort
    idx_K_LaS  = np.zeros(vx.shape);
    idx_K_LaS  [ss[0:nk]] = 1;
    return idx_K_LaS 
##############################################################################################################
#path of test sample
input_path='PATH TO legitimate sample'
output_path='PATH TO Output Folder to SAVE Adversarial Example'
classes=['spyder','dog','cat','squirrel','sheep','butterfly','horse','elephant','cow','chicken']
#link of public dataset:  https://www.kaggle.com/alessiocorrado99/animals10
# In this example,We input an image with a 'chicken' label,
#if the input sample has another label, the argument of 'classes[9]' should be updated accordingly.  
original_label =classes[9]; 
list0 = [];listName0 = []; 
list0,listName0 = load_images_from_folder(input_path)
##############################################################################################################
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
# Get input and output tensors.
input_details = interpreter.get_input_details();print()
output_details = interpreter.get_output_details();
print(output_details)
input_shape = input_details[0]['shape']
##############################################################################################################
#allList = list0[0:100] +list1[0:100] +list2[0:100] +list3[0:100] +list4[0:100] +list5[0:100] +list6[0:100] +list7[0:100] +list8[0:100]+ list9[0:100]    
allList = list0 
k=16; #number of nonzeros 
alpha = 0.001 # MSE
QueryNumber = 1000; #number of query
L =len(allList);
ctr_k = 0;ctr_l = 0;ctr_a = 0;
idx_k = np.zeros((L,1));

for idx in range(L):
    img1 = allList[idx];
    legitimateImg=np.zeros(img1.shape,np.uint8);
    legitimateImg[:,:,0]=img1[:,:,2];legitimateImg[:,:,1]=img1[:,:,1];legitimateImg[:,:,2]=img1[:,:,0];
    legImg = resize(legitimateImg, (224, 224));#img = img/255
    
    # Testing the model to observe if it initisally predicts the legitimate sample correctly
    input_data = (255*np.expand_dims(legImg, axis=0)).astype(np.uint8);
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data_leg = interpreter.get_tensor(output_details[0]['index'])
    pro_leg  = (output_data_leg[0][[output_data_leg.argmax(axis=-1)[0]]][0]) / np.sum(output_data_leg)
    label = classes[output_data_leg.argmax(axis=-1)[0]];
    

    if label!=original_label:
        print('This sample has already been misclassified by the model---Try another sample');
        continue;
        
################################ adding noise to the k largest components in sparse domain ##############
    #DCT Transformation
    yd1 = dct(legImg, axis=0, norm="ortho");
    yd2 = dct(yd1, axis=1, norm="ortho");
    idx_K_LaS   = Klargest(yd2,k) # position of K LaS components 
    model_fooled = False;
    
    for q in range(QueryNumber):
        y = np.random.normal(0,1,legImg.shape)
        yd1 = dct(y, axis=0, norm="ortho");
        yd2 = dct(yd1, axis=1, norm="ortho");        
        if not model_fooled:            
            yd3 = np.zeros(yd2.shape);
            temp = yd2.reshape(-1);
            yd3 = (np.multiply(temp,idx_K_LaS )).reshape(224,224,3)            
            #inverse DCT Transformation
            iyd1 = idct(yd3, axis=1, norm="ortho");
            iyd2 = idct(iyd1, axis=0, norm="ortho")           
            yLF = np.sqrt(alpha) * iyd2 / np.sqrt(np.square(iyd2).mean(axis=None))#limited noise
            xAdvKlargest = np.add(yLF,legImg)# adding limited noise to the legitimate sample     
            MSE = (np.square(yLF)).mean(axis=None); #print(MSE)
############sending Query to the the TFLite model trained by Google Cloud Vision for evaluation
            input_data = (255*np.expand_dims(xAdvKlargest, axis=0)).astype(np.uint8);
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])
            pro  = (output_data[0][[output_data.argmax(axis=-1)[0]]][0]) / np.sum(output_data)
            if classes[output_data.argmax(axis=-1)[0]]!= label:
                print('================================================');
                print('------The model is fooled in query number: ', q);
                print('legitimate predicted label was:  ', classes[output_data_leg.argmax(axis=-1)[0]],'    ', pro_leg)
                print(output_data_leg)

                print(' Adversarial predicted label is:  ', classes[output_data.argmax(axis=-1)[0]],'    ', pro )
                print(output_data)
                print("************************")
                model_fooled = True;
                
                temp_img=np.zeros(xAdvKlargest.shape);
                temp_img[:,:,0]=xAdvKlargest[:,:,2];temp_img[:,:,1]=xAdvKlargest[:,:,1];temp_img[:,:,2]=xAdvKlargest[:,:,0];
                cv2.imwrite(output_path+'adv.jpg', 255*temp_img)
                break
            else:
                print("query number: " + str(q) +' unsuccessful!')


##############################################################################################################

plt.imshow(legImg);plt.title('Legitimate image: '+ classes[output_data_leg.argmax(axis=-1)[0]],);plt.show()
plt.imshow(xAdvKlargest); plt.title('Adversarial image: '+classes[output_data.argmax(axis=-1)[0]]);plt.show()
