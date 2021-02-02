import RPI.GPIO as GPIO
import time

GPIO.setmode(GPIO.BOARD)
GPIO.setup(7, GPIO.OUT)
GPIO.setup(11, GPIO.OUT)
GPIO.setup(13, GPIO.OUT)
GPIO.setup(15, GPIO.OUT)

GPIO.output(7, True)
GPIO.output(11, True)
GPIO.output(13, True)
GPIO.output(15, True)

time.sleep(0.5)       #action takes 0.5 sec

gpio.cleanup()
