import RPI.GPIO as GPIO
import time


GPIO.setmode(GPIO.BCM)          #start the setmode

TRIG = 4                        #fixe the TRIG on the pin 4
ECHO = 18                       #fixe the ECHO on the pin 18

#we set up both the trig and the echo
GPIO.setup(TRIG, GPIO.OUT)      
GPIO.setup(ECHO, GPIO.IN)

def get_distance():

	#the trig send the sound
	GPIO.output(TRIG, True)
	#suspend the execution for this "0.0001" seconds
	time.sleep(0.0001)

	#the trig stop sending sound
	GPIO.output(TRIG, False)

	while GPIO.input(ECHO) == False:      #while we have not receive the echo 
		start = time.time()               #start counting time
	while GPIO.input(ECHO) == True:       #while receiving the echo
		end = time.time()

	sig_time = end-start                 

	distance = sig_time/0.000058          #we gonna use the v=d/t (vitesse du son)

	print('Distance : {} cm '.format(distance) )

	GPIO.cleanup()           #we clean all the pins
	return distance

