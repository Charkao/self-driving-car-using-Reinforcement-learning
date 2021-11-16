import serial
import numpy as np
import cv2
from skimage import io , color,transform
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
import matplotlib.pyplot as plt
import pickle
from keras.optimizers import Adam
from keras.optimizers import SGD 
import random
import time
import os
import picamera
from time import sleep
from skimage import io , color,transform
import numpy as np
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.layers import Flatten, Conv2D, MaxPooling2D
import pickle
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.initializers import Zeros
import random
import time
import os
import matplotlib.pyplot as plt
#### importation des bibliothéques utiles : kerras pour le deep learning , opencv, et skimage pour le traitement d'images
epsilon_greedy=0
gamma = 0.9
##   epsilongreedy est le pourcentage d'actions aléatoires et gamma le facteur de dépreciation de l'algorithme d'apprentissage
if (os.path.isdir("model_apprentissage_reel_CNN_forward") ):
    print("model existant")
    model=load_model("model_apprentissage_reel_CNN_forward")

    
else :   
    #CHANGE IT TO CNN
    model = Sequential()
    model.add( Conv2D(64,(3,3),input_shape=(100,100,3) ) )
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Conv2D(64,(3,3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))

    model.add(Flatten())
    model.add(Dense(64))

    model.add(Dense(4))
    model.add(Activation("linear"))

    model.compile(loss='mse',optimizer=Adam(lr=0.001),metrics=['accuracy'])
######création des couches du réseaux de neurones de type CNN 

ser = serial.Serial('/dev/ttyACM0', 9600) ##liason série entre la raspberry et l'arduino 



#################################################################################################
###Fonctions necessaires à l'apprentissage
def train(state, action, reward, next_state, done):##fonction qui entraine le réseaux de neurone
    target = model.predict(np.array([state]))[0]
    if done:
        target[action] = reward
    else:
        target[action] = reward + gamma * np.max(model.predict(np.array([next_state])))
    inputs = np.array([state])
    outputs = np.array([target])
    return model.fit(inputs,outputs,epochs=1,verbose=0,batch_size=1)
    
    
def get_best_action(state, rand=True):##fonction qui donne l'action predite par le RNN ou une action aléatoire
    if rand and np.random.rand() <= epsilon_greedy:
        print("aleatoire")
        a=random.randrange(0,3)
        L2 = []
        if(a==0):
            L2=[8,0]
            return L2
        elif (a==1):
            L2=[4,1]
            return L2
        elif(a==2):
            L2=[5,2]
            return L2
        else :
            L2=[6,3]
            return L2
    act_values = model.predict(np.array([state]))
    # print(act_values)
    action =  np.argmax(act_values[0])
    # print("action predite est ",action)
    # print("non aléatoire")
    L = []
    if(action==0):
        L=[8,0]
        return L
    elif (action==1):
        L=[4,1]
        return L
    elif(action==2):
        L=[5,2]
        return L
    else :
        L=[6,3]
        return L


def distance() :##fonction qui mesure la distance entre l'objectif et le robot
    cap = cv2.VideoCapture(0)
    l0 = np.array([90,100,50])
    l1 =  np.array([120,255,255]) 
    i=0
    while True:
            i+=1
            ret , frame = cap.read()  
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(image, l0, l1)
            mask = cv2.erode(mask,None,iterations=4)
            mask = cv2.dilate(mask,None,iterations=4)
            image2 = cv2.bitwise_and(frame, frame,mask=mask)
            elements = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
            d=-1
            cv2.imwrite("state.png",mask)
            
            if (len(elements)==2 ): 
                ((x2,y2),rayon2) = cv2.minEnclosingCircle(elements[1])
                ((x1,y1),rayon1) = cv2.minEnclosingCircle(elements[0])
                distance = np.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))*(26/104)
                print("le robot est la  , la distance est de ",distance)
                d = distance
            if(len(elements)==3):
                sorted(elements,key=cv2.contourArea)
                ((x2,y2),rayon2) = cv2.minEnclosingCircle(elements[1])
                ((x1,y1),rayon1) = cv2.minEnclosingCircle(elements[2])
                distance = np.sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1))*(26/104)
                print("le robot est la  , la distance est de ",distance)
                d = distance
            print("sortie de zone du robot")
            if i==3:
                 break
    cap.release()
    cv2.destroyAllWindows()
    return d

def get_reward():###fonction qui donne la récompense au robot lors de la transition d'état
    d=distance()
    frame=cv2.imread("state.png")
    state = transform.resize(frame, (100,100))
    sortie  = False
    arrive = False 
    if(d==-1):
        return -10,False,True,state 
    
    if(d<40):
       return 10 , True , False,state 
    
    return 0,False,False,state

    
def move (action ):##fonction qui envoie l'ordre à l'arduino pour exécuter une action
    action=str(action)
    a = str.encode(action)
    time.sleep(2)
    ser.write(a)
    

def move_robot(action):
    # avancer le robot
    move(action)
    # obtenir la recompense 
    reward,done,finish,state = get_reward()
    return reward,done,finish,state
    
def get_init_img():##fonction qui recupére l'image de l'environnement à partir de la camera du haut
    distance()
    frame=cv2.imread("state.png")
    state = transform.resize(frame, (100,100))
    return state

def inverse_action(action):####fonction qui inverse les actions , elle servira à ramenener le Robot à sa position initiale  
    if(action == 8):
        return  5
    elif ( action == 5):
        return  8
    elif ( action == 6):
        return 4
    elif (action == 4):
        return 6
    else:
        print('action non inversible')
        return 1


######################################################################################################  
###Boucle de l'Apprentissage
nb_episodes = 100
for i in range (nb_episodes):
    scores = []
    finish = False
    score = 0
    state = get_init_img()
    step = 0
    Liste_retour=[]
    L=[]
    while not finish :
        step+=1
        L = get_best_action(state)
        action1 = L[0]
        action2 = L[1]
        Liste_retour.append(action1)
        reward,done ,finish,next_state = move_robot(action1)
        print("reward est: ", reward)
        score=score+reward
        train(state, action2, reward, next_state, done)
        state = next_state
        if done:
            print("Felicitations c'est fini")
            scores=scores+[score]
            break
        if finish:
            print("je suis sortie")
            scores=scores+[score]
            break
        if step > 6:
            train(state, action2, -10, state, True)
            print("j'ai bcp essayé")
            scores=scores+[score]
            break
    ##retour du robot par inversion des actions
    request = input('enter a return back request  / [1/0]')
    if(request=='1'):
        for j in range(len(Liste_retour)):
            action1 = int ( Liste_retour[len(Liste_retour)-j-2] )
            action = inverse_action(action1)
            action= str(action)
            action2 = str.encode(action)
            ser.write(action2)
            time.sleep(2)
    
model.save("model_apprentissage_reel_CNN",overwrite=True)###sauvegarde des poids du CNN
####affichage  de la courbe de performance
plt.figure(1)
t=[i for i in range(4)]
plt.plot(t,scores)
plt.show()
