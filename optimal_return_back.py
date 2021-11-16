#!/usr/bin/env python3
# -*- coding: utf-8 -*-



#ce code permet de faire retourner le robot vers sa position initale de mannière autonomne suivant un chemin optimal 

"""
Created on Fri Apr  9 21:07:26 2021

@author: charkaoui
"""
import serial
import numpy as np
import cv2
import skimage
from skimage import io 
import time
ser = serial.Serial('/dev/ttyACM0', 9600)


#cette fonction permet de renvoyer les coordonnées du robot et son orientation

def where():
    #red : 0--10
    cap = cv2.VideoCapture(0)
    l0 = np.array([90,100,50])
    l1 =  np.array([120,255,255]) 
    i=0
    while True:
            i+=1
            ret , frame = cap.read()  
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            # image = cv2.blur(image,(7,7))
            mask = cv2.inRange(image, l0, l1)
            mask = cv2.erode(mask,None,iterations=4)
            mask = cv2.dilate(mask,None,iterations=4)
            image2 = cv2.bitwise_and(frame, frame,mask=mask)
            elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            d=-1
            a= True
            ang = 0
            if (len(elements)==1   ):
                sorted(elements,key=cv2.contourArea)          
                ((x2,y2),rayon2) = cv2.minEnclosingCircle(elements[0])
                (x,y) , (ma,a) , angle = cv2.fitEllipse(elements[0])
                cv2.circle(image2, (int(x2),int(y2)), int(rayon2), (0,255,0))
                ((x1,y1),rayon1) = cv2.minEnclosingCircle(elements[0])
                cv2.circle(image2, (int(x1),int(y1)), int(rayon1), (0,255,0))
                distance = np.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))*30/211
                d = distance
                a= False
                ang = angle
            cv2.imshow('camera',frame)
            cv2.imshow('image2',image2)
            cv2.imshow('mask',mask)
            cv2.imwrite("ali.jpg",image2)
            if i==3:
                break
    cap.release()
    cv2.destroyAllWindows()
    return ang,y2,x2


#postion initale de coordonées: (x=y2=168, y=x2=628)


#cette fonction permet de faire déplacer le robot vers sa position initale
def retour_inverse():
    x=168
    y=600
    angle,x1,y1=where()
    print("angle = ",angle)
    while(np.abs(angle-90)>20):
        print("this is angle",angle)
        angle,x1,y1=where()
        if(angle-90>0):
            action = str.encode("2")
            ser.write(action)
            time.sleep(2)
        if(angle-90<0):
            action = str.encode("1")
            ser.write(action)
            time.sleep(2)
   
    while(np.abs(y1-y)>50):
        angle,x1,y1=where()
        print("x1=",x1,"y1",y1)
        if(y1-y>0):
            action = str.encode("7")
            ser.write(action)
            time.sleep(2)
        if(y1-y<0):
            action = str.encode("3")
            ser.write(action)
            time.sleep(2)
    if(x1>x):
        action = str.encode("6")
        ser.write(action)
        time.sleep(2)
        while(np.abs(x1-x)>50):
            angle,x1,y1=where()
            action = str.encode("7")
            ser.write(action)
            time.sleep(2)
        action = str.encode("4")
        ser.write(action)
        time.sleep(2)
    else:
        action = str.encode("4")
        ser.write(action)
        time.sleep(2)
        while(np.abs(x1-x)>50):
            angle,x1,y1=where()
            action = str.encode("7")
            ser.write(action)
            time.sleep(2)
        action = str.encode("6")
        ser.write(action)
        time.sleep(2)
        


                
        
        
            
            
    
        
        
            
    
