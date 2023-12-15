
#!/usr/bin/env python3.9

# Version 1.0.0

from time import sleep
from threading import Timer
import cv2

try:
    import RPi.GPIO as GPIO
    RPi_GPIO_present = True
except ImportError:
    RPi_GPIO_present = False
    
# Stepper motor settings
DIR = 27   # Direction GPIO Pin
STEP = 22  # Step GPIO Pin
CW = 1     # Clockwise Rotation
CCW = 0    # Counterclockwise Rotation
STEPON = 0  # pin to turn on/off the stepper

# buttons
btn_left = 13
btn_right = 19
btn_start = 16
btn_stop = 26
btn_rewind = 20

photoint = 21  # photointeruptor

ledon = 12  # pin for LED
pin_forward = 6  # motor pin (spool)
pin_backward = 5

delay = .005  # delay inbetween steps
tolstep = 2 // 2  # defines how many steps are done for correction
steps = 0


step_minus = 0  # counter for stepper corrections
step_plus = 0
rewind = 0

spool_pwm = None
led_pwm = None

led_dc = 100
    
def initGpio() :
    global spool_pwm
    global led_pwm
    
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(DIR, GPIO.OUT)
    GPIO.setup(STEP, GPIO.OUT)
    GPIO.setup(STEPON, GPIO.OUT)
    GPIO.setup(ledon, GPIO.OUT)
    GPIO.setup(pin_forward, GPIO.OUT)
    GPIO.setup(pin_backward, GPIO.OUT)
    GPIO.setup(btn_right, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(btn_left, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(btn_start, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(btn_stop, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(btn_rewind, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup(photoint, GPIO.IN, pull_up_down=GPIO.PUD_UP)
    GPIO.setup((18, 15, 14), GPIO.OUT)

    #GPIO.output(ledon, GPIO.HIGH)  # turn on LED
    
    spool_pwm = GPIO.PWM(pin_forward, 40)  # set PWM channel, hz
    led_pwm = GPIO.PWM(ledon, 40)
    led_pwm.start(led_dc)

def setLedDc(dc):
    global led_dc
    led_dc = dc
    led_pwm.ChangeDutyCycle(led_dc)

def ledPlus():
    global led_dc
    led_dc+=10
    if led_dc>100:
        led_dc=100
    led_pwm.ChangeDutyCycle(led_dc)
    return led_dc
    
def ledMinus():
    global led_dc
    led_dc-=10
    if led_dc<0:
        led_dc=0
    led_pwm.ChangeDutyCycle(led_dc)
    return led_dc

def spoolFwd(time=1):
    print("Spool forward")
    spoolTimer = Timer(time, spoolStop)
    spoolTimer.start()
    pint = GPIO.input(photoint)
    if pint:
        GPIO.output(pin_forward, GPIO.HIGH)
        GPIO.output(pin_backward, GPIO.LOW)
        spool_pwm.start(10)
    else:
        spool_pwm.ChangeDutyCycle(0)

def spoolBack(time=1):
    print("Spool back")
    spoolTimer = Timer(time, spoolStop)
    spoolTimer.start()
    GPIO.output(pin_forward, GPIO.LOW)
    GPIO.output(pin_backward, GPIO.HIGH)
    spool_pwm.start(10)
    
def spoolStop():
    print("Spool stop")
    spool_pwm.ChangeDutyCycle(0)
    GPIO.output(pin_forward, GPIO.LOW)
    GPIO.output(pin_backward, GPIO.LOW)
    spool_pwm.ChangeDutyCycle(0)

def rewind():
    spool_pwm.ChangeDutyCycle(50)
    GPIO.output(pin_forward, GPIO.HIGH)
    GPIO.output(pin_backward, GPIO.LOW)

def stepHigh():
    GPIO.output(STEP, GPIO.HIGH)
    Timer(delay, stepLow).start()

def stepLow():
    GPIO.output(STEP, GPIO.LOW)

def adjDn():
    delay = 0.005
    steps = 6
    GPIO.output(DIR, CW)
    for x in range(steps):
        Timer(delay, stepHigh).start()
        
def stepCw(steps):
    #delay = 0.005
    print(f"steps cw {steps} delay {delay}")
    GPIO.output(DIR, CW)
    for x in range(steps):
        #Timer(delay, stepHigh).start()
        GPIO.output(STEP, GPIO.HIGH)
        sleep(delay)
        GPIO.output(STEP, GPIO.LOW)
        sleep(delay)

def stepCcw(steps):
    #delay = 0.005
    print(f"steps ccw {steps} delay {delay}")
    GPIO.output(DIR, CCW)
    for x in range(steps):
        #Timer(delay, stepHigh).start()
        
        GPIO.output(STEP, GPIO.HIGH)
        sleep(delay)
        GPIO.output(STEP, GPIO.LOW)
        sleep(delay)

def stopScanner():
    spoolStop()
    GPIO.output(ledon, GPIO.LOW)
    GPIO.output(STEPON, GPIO.LOW)

def startScanner():
    GPIO.output((18, 15, 14), (1, 1, 0))
    #GPIO.output(ledon, GPIO.HIGH)  # turn on LED
    setLedDc(100)
    GPIO.output(STEPON, GPIO.HIGH)
    sleep(0.5)

def cleanup():
    stopScanner()
    GPIO.cleanup()
 
