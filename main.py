from sense_hat import SenseHat
from pathlib import Path 
import numpy as np
from PIL import ImageFilter, Image
import cv2
import time
import threading
import argparse

import numpy as np
import tensorflow as tf

class MemImage:
    def __init__(self):
        self.bytes = bytearray()
    def write(self, new_bytes):
        self.bytes.extend(new_bytes)
    def get_bytes(self):
        return bytes(self.bytes)
        

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  format = ""
  
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(file_reader, name="png_reader")
    format = "png"
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
    format = "gif"
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    format = "bmp"
  elif file_name.endswith(".jpg"):
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
    format = "jpg"

  return tensor_from_image(file_reader, format, input_height=input_height, input_width=input_width, input_mean=input_mean, input_std=input_std)

def tensor_from_image(image, format,
                      input_height=299,
                      input_width=299,
                      input_mean=0,
                      input_std=255):
  if format == "png":
    image_reader = tf.image.decode_png(
        image, channels=3, name="png_reader")
  elif format == "gif":
    image_reader = tf.squeeze(
        tf.image.decode_gif(image, name="gif_reader"))
  elif format == "bmp":
      image_reader = tf.image.decode_bmp(image, name="bmp_reader")
  elif format == "jpg":
    image_reader = tf.image.decode_jpeg(
        image, channels=3, name="jpeg_reader")
  else:
    print("Unknow format when in tensor_from_image")

  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

class WildfireRecognizer:
    def __init__(self, model_file = "model.pb"):
        self.graph = load_graph(model_file)
        self.labels = ['fire','no fire' ]


    def recognize_tensor(self, tensor):
      input_layer = "Placeholder"
      output_layer = "final_result"
      input_name = "import/" + input_layer
      output_name = "import/" + output_layer
      input_operation = self.graph.get_operation_by_name(input_name)
      output_operation = self.graph.get_operation_by_name(output_name)

      with tf.Session(graph=self.graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: tensor
        })
      results = np.squeeze(results)
      #Fire probability
      return results[0]
    
    def recognize_from_file(self, image_path):
      tensor = read_tensor_from_image_file(
          image_path,
          input_height = 299,
          input_width = 299,
          input_mean = 0,
          input_std = 255)
      return self.recognize_tensor(tensor)
    
    def recognize_from_image_data(self, image_data, format):
      tensor = tensor_from_image(image_data, format)
      return self.recognize_tensor(tensor)


'''
AsPi helper code 

taken from https://github.com/coderdojo-futurix/aspi)
Copyright (c) CoderDojo Futurix <coderdojo@futurix.pt>
Licensed under MIT License
'''

from gpiozero import CPUTemperature
from datetime import datetime 
from logging import Formatter
from logzero import logger
from pisense import SenseHAT
from ephem import readtle, degrees
from threading import Timer, Thread
from queue import Queue
from collections import OrderedDict
import logzero
import os
import time
import locale
import math
import sys
import signal
import picamera
import picamera.array
import numpy as np

# Global AstroPi device objects
sense_hat = SenseHAT()
cpu = CPUTemperature()
camera = picamera.PiCamera()

# Default values
MIN_LOG_PERIOD_IN_SECS = 2
MIN_IMG_PERIOD_IN_SECS = 5
SHUTDOWN_TIMEOUT_IN_SECS = 3 * 60
DEFAULT_DURATION_IN_SECS = 3 * 60 * 60 - SHUTDOWN_TIMEOUT_IN_SECS
DEFAULT_SIZE_PER_LOGFILE_IN_BYTES = 30*1024*1024
DEFAULT_LOG_PERIOD_IN_SECS = 5
DEFAULT_IMG_PERIOD_IN_SECS = 10
DEFAULT_LOGFILE_PREFIX = "sense_hat_logger"
PICAMERA_SENSOR_MODE_2_RESOLUTION = ( 2592, 1944 )
ASTROPI_ORIENTATION = 270
ENABLE_DEBUG = False
NO_READING=-1
LOG_FORMAT='%(asctime)-15s.%(msecs)03d,%(message)s'
DATE_FORMAT='%Y-%m-%d %H:%M:%S'        
TIMESTAMP_FIELD = "timestamp"
TIMESTAMP_FORMAT='{:%Y-%m-%d_%Hh%Mm%Ss}'
DEGREES_PER_RADIAN = 180.0 / math.pi
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
LOGFILE_EXT = 'csv'
NO_TIMESTAMP_FORMATTER = Formatter()


def get_timestamp():
    '''
    Simple method to return a formatted timestamp of the current time
    '''
    return TIMESTAMP_FORMAT.format(datetime.now())

class AsPi:
    '''
    Utility class that holds constants for the sensors names and respective values units
    and manages the end of the program 
    '''
    # Flag to indicate the program has ended
    _ended = False

    # Holds the termination timer after the end is called
    shutdowntimer = None


    def isShuttingDown():
        '''
        Indicates if the program has ended and the termination timer is active
        '''
        return not AsPi.shutdowntimer is None

    def terminate():
        '''
        Self terminates by issuing a SIGTERM to itself
        '''
        os.kill(os.getpid(),signal.SIGTERM)

    def end():
        '''
        Call to gracefully end of the program. A termination timer is also started. 
        If the program cleanup actions are not done in SHUTDOWN_TIMEOUT_IN_SECS seconds, 
        self termination is issued.
        '''
        if not AsPi.hasEnded():
            AsPi._ended = True
            print("Forcing termination in " + str(SHUTDOWN_TIMEOUT_IN_SECS) + " secs")
            AsPi.shutdowntimer = Timer( SHUTDOWN_TIMEOUT_IN_SECS , AsPi.terminate )
            AsPi.shutdowntimer.start()

    def hasEnded():
        '''
        Indicates if the program ended
        '''
        return AsPi._ended

    # Sensor names constants
    SENSOR_CPU_TEMP = "cpu_temp"
    SENSOR_TEMPERATURE = "temperature"
    SENSOR_PRESSURE = "pressure"
    SENSOR_HUMIDITY = "humidity"
    SENSOR_COMPASS_X = "compass_x"
    SENSOR_COMPASS_Y = "compass_y"
    SENSOR_COMPASS_Z = "compass_z"
    SENSOR_GYRO_X = "gyro_x"
    SENSOR_GYRO_Y = "gyro_y"
    SENSOR_GYRO_Z = "gyro_z"
    SENSOR_ACCEL_X = "accel_x"
    SENSOR_ACCEL_Y = "accel_y"
    SENSOR_ACCEL_Z = "accel_z"
    SENSOR_PITCH = "pitch"
    SENSOR_ROLL = "roll"
    SENSOR_YAW = "yaw"
    SENSOR_LAT = "lat"
    SENSOR_LON = "long"
    SENSOR_ELEVATION = "elevation"
    SENSOR_ECLIPSED = "eclipsed"
    SENSOR_MOTION = "motion"
    SENSOR_USERDATA = "userdata"

    # Units constants
    UNITS_DEGREES_CELSIUS = "°C"
    UNITS_RADIANS = "rad"
    UNITS_RADIANS_PER_SEC = UNITS_RADIANS + "/sec"
    UNITS_STANDARD_GRAVITIES = "g"
    UNITS_MICRO_TESLAS = "uT"
    UNITS_MILLIBARS = "mbar"
    UNITS_PERC_RELATIVE_HUMIDITY = "%RH"
    UNITS_DEGREES = "°"
    UNITS_METERS = "m"
    UNITS_BOOL = "bool"
    UNITS_COUNT = "n"
    UNITS_STR = "str"

    # Units of the values reported by each sensor
    UNITS = OrderedDict( [
        ( SENSOR_CPU_TEMP    , UNITS_DEGREES_CELSIUS        ) ,
        ( SENSOR_TEMPERATURE , UNITS_DEGREES_CELSIUS        ) ,
        ( SENSOR_PRESSURE    , UNITS_MILLIBARS              ) ,
        ( SENSOR_HUMIDITY    , UNITS_PERC_RELATIVE_HUMIDITY ) ,
        ( SENSOR_COMPASS_X   , UNITS_MICRO_TESLAS           ) ,
        ( SENSOR_COMPASS_Y   , UNITS_MICRO_TESLAS           ) ,
        ( SENSOR_COMPASS_Z   , UNITS_MICRO_TESLAS           ) ,
        ( SENSOR_GYRO_X      , UNITS_RADIANS_PER_SEC        ) ,
        ( SENSOR_GYRO_Y      , UNITS_RADIANS_PER_SEC        ) ,
        ( SENSOR_GYRO_Z      , UNITS_RADIANS_PER_SEC        ) ,
        ( SENSOR_ACCEL_X     , UNITS_STANDARD_GRAVITIES     ) ,
        ( SENSOR_ACCEL_Y     , UNITS_STANDARD_GRAVITIES     ) ,
        ( SENSOR_ACCEL_Z     , UNITS_STANDARD_GRAVITIES     ) ,
        ( SENSOR_PITCH       , UNITS_RADIANS                ) ,
        ( SENSOR_ROLL        , UNITS_RADIANS                ) ,
        ( SENSOR_YAW         , UNITS_RADIANS                ) ,
        ( SENSOR_LAT         , UNITS_DEGREES                ) ,
        ( SENSOR_LON         , UNITS_DEGREES                ) ,
        ( SENSOR_ELEVATION   , UNITS_METERS                 ) ,
        ( SENSOR_ECLIPSED    , UNITS_BOOL                   ) ,
        ( SENSOR_MOTION      , UNITS_COUNT                  ) , 
        ( SENSOR_USERDATA    , UNITS_STR                    )
    ])

    # list with all sensor names
    ALL_SENSORS = UNITS.keys()


class AsPiResult:
    '''
    Class that stores one and only one value to safely exchanve values between threads
    '''
    def __init__(self):
        self.result = Queue(maxsize = 1)
    
    def put( self, data ):
        try:
            self.result.get_nowait()
        except:
            pass
        finally:
           self.result.put( data )

    def get( self , timeout=1):
        data = None
        try:
            data = self.result.get( timeout )
        except:
            pass
        finally:
            if AsPi.hasEnded():
                return None
            return data

class AsPiSensors:
    '''
    Class that makes takes measurements from all sensors 
    '''
    userData = AsPiResult()
    lastAsPiSensorsReading = AsPiResult()
    cpu = CPUTemperature()
    iss = readtle(
            'ISS (ZARYA)' ,
            '1 25544U 98067A   19027.92703822  .00001504  00000-0  30922-4 0  9992',
            '2 25544  51.6413 338.8011 0004969 323.5710 139.9801 15.53200917153468'
        )

    def __init__(self, hat, selected_sensors = AsPi.ALL_SENSORS ):
        self.hat = hat
        self.selected_sensors = selected_sensors        
        
    def _get_latlon():
        AsPiSensors.iss.compute()
        lat = AsPiSensors.iss.sublat * DEGREES_PER_RADIAN
        lon = AsPiSensors.iss.sublong * DEGREES_PER_RADIAN
        alt = AsPiSensors.iss.elevation
        ecl = AsPiSensors.iss.eclipsed
        return lat,lon, alt, ecl

    def read(self):
        
        envreading = self.hat.environ.read()
        imureading = self.hat.imu.read()
        latitude, longitude, elevation, eclipsed = AsPiSensors._get_latlon()

        self.readings = { 
            AsPi.SENSOR_CPU_TEMP : AsPiSensors.cpu.temperature,
            AsPi.SENSOR_TEMPERATURE : envreading.temperature, 
            AsPi.SENSOR_PRESSURE : envreading.pressure,
            AsPi.SENSOR_HUMIDITY : envreading.humidity,
            AsPi.SENSOR_COMPASS_X : imureading.compass.x ,
            AsPi.SENSOR_COMPASS_Y : imureading.compass.y ,
            AsPi.SENSOR_COMPASS_Z : imureading.compass.z ,
            AsPi.SENSOR_GYRO_X : imureading.gyro.x ,
            AsPi.SENSOR_GYRO_Y : imureading.gyro.y ,
            AsPi.SENSOR_GYRO_Z : imureading.gyro.z ,
            AsPi.SENSOR_ACCEL_X : imureading.accel.x ,
            AsPi.SENSOR_ACCEL_Y : imureading.accel.y ,
            AsPi.SENSOR_ACCEL_Z : imureading.accel.z ,
            AsPi.SENSOR_PITCH : imureading.orient.pitch ,
            AsPi.SENSOR_ROLL : imureading.orient.roll,
            AsPi.SENSOR_YAW : imureading.orient.yaw,
            AsPi.SENSOR_LAT : latitude,
            AsPi.SENSOR_LON : longitude,
            AsPi.SENSOR_ELEVATION : elevation,
            AsPi.SENSOR_ECLIPSED : eclipsed,
            AsPi.SENSOR_MOTION : MotionAnalyser.occurrences, 
            AsPi.SENSOR_USERDATA :  AsPiSensors.userData.get( timeout=0)
      
        }

        # Reset motion detection occurences count
        MotionAnalyser.occurrences = 0

        AsPiSensors.lastAsPiSensorsReading.put( self.readings )

        # Return readings from selected sensors only
        return self.__selected_sensors_only( self.readings )

    def __selected_sensors_only(self, readings ):
        return list( map( lambda sensor: readings[ sensor ],  self.selected_sensors ))


class AsPiTimer:
    '''
    Recurrent Timer. It's the same as python threading timer class but a recurring one.
    Everytime it fires, calls the callback and sets another Timer. It keeps doing that until 
    it's cancelled.   
    '''
    def __init__( self, periodInSecs=DEFAULT_LOG_PERIOD_IN_SECS, func=None ):
        self.periodInSecs = periodInSecs
        self.func = func
        self.__create_timer()

    def __create_timer(self):
        self.aspitimer = Timer( self.periodInSecs, self.func )

    def start(self):
        self.__create_timer()
        self.aspitimer.start()
    
    def reset(self):
        self.start()

    def cancel(self):
        self.aspitimer.cancel()

class AsPiMemImage:
    '''
    Class to store an image in memory. 
    To be compatible with Tensorflow classification functions.
    '''
    def __init__(self):
        self.bytes = bytearray()
    def write(self, new_bytes):
        self.bytes.extend(new_bytes)
    def get_bytes(self):
        return bytes(self.bytes)


class MotionAnalyser( picamera.array.PiMotionAnalysis ):
    '''
    Analyses frames from recording video and checks if the
    frame vectors cross the thresholds indicating that movement
    was detected or not. If it detects movement the occurrences
    variable is incremented and it can be queried for movement
    events
    '''
    occurrences=0

    def analyse(self, a):
        a = np.sqrt(
            np.square(a['x'].astype(np.float)) +
            np.square(a['y'].astype(np.float))
            ).clip(0, 255).astype(np.uint8)
        # If there're more than 10 vectors with a magnitude greater
        # than 60, then say we've detected motion
        if (a > 60).sum() > 10:
            MotionAnalyser.occurrences += 1

class AsPiMotionDetector( Thread ):
    '''
    Starts and stops camera recording to /dev/null sending
    the frames to the MotionAnalyser class to detect movement.
    '''
    def __init__(self):
        Thread.__init__(self)

    def run(self):
        AsPiMotionDetector.start_motion_detection()
        while True:
            if AsPi.hasEnded():
                self.stop()
                return
            time.sleep(1)

    def stop(self):
        AsPiMotionDetector.stop_motion_detection()

    def start_motion_detection():
        camera.resolution = (640, 480)
        camera.framerate = 30
        camera.start_recording(
            '/dev/null', format='h264',
            motion_output = MotionAnalyser( camera )
            )

    def stop_motion_detection():
        if camera.recording:
            camera.stop_recording()


class AsPiCamera( Thread ):
    '''
    If enabled (it's disabled by default), it starts a thread, periodically taking pictures 
    with the AstroPi camera and storing them in files, putting the current ISS position in 
    the image file EXIF tags. It also stores the image in the lastPictureTaken class variable.
    '''
    lastPictureTaken = AsPiResult()
    isCameraEnabled = False
    
    def __init__(self, imgPeriodInSecs ):
        Thread.__init__(self)
        self.imgtimer = AsPiTimer( periodInSecs=imgPeriodInSecs, func=self.__take_picture )

    def run(self):
        self.imgtimer.start()
        while True:
            if AsPi.hasEnded():
                self.stop()
                return
            time.sleep(1)

    def stop(self):
        self.imgtimer.cancel()

    def __take_picture(self):
        AsPiMotionDetector.stop_motion_detection()
        self.__capture_image()
        AsPiMotionDetector.start_motion_detection()

    def __save_image( img, imgfilename ):
        imgfile = open(imgfilename, 'wb')
        imgfile.write( img.get_bytes() )
        imgfile.close()

    def __capture_image(self):
        imgfileprefix = AsPiLogFile.generate_fileprefix() 
        camera.resolution = PICAMERA_SENSOR_MODE_2_RESOLUTION
        camera.exif_tags['Artist'] = imgfileprefix
        lat,lon = self.__set_latlon_in_exif()
        imgfilename = imgfileprefix + "_" + lat + "_" + lon + ".jpg"
        img = AsPiMemImage()
        camera.capture( img , "jpeg")
        AsPiCamera.lastPictureTaken.put( ( imgfilename, img ) )
        AsPiCamera.__save_image( img, imgfilename )

        if AsPi.hasEnded():
            self.imgtimer.cancel()
        else:
            self.imgtimer.reset()

    def __set_latlon_in_exif(self):
        """
        A function to write lat/long to EXIF data for photographs
        (source based in the get_latlon function available in the 2019 AstroPi Mission 
        SpaceLab Phase 2 guide in the "Recording images using the camera" section)
        """
        AsPiSensors.iss.compute() # Get the lat/long values from ephem
        long_value = [float(i) for i in str(AsPiSensors.iss.sublong).split(":")]
        if long_value[0] < 0:
            long_value[0] = abs(long_value[0])
            longitude_ref = "W"
        else:
            longitude_ref = "E"
        longitude = '%d/1,%d/1,%d/10' % (long_value[0], long_value[1], long_value[2]*10)
        lat_value = [float(i) for i in str( AsPiSensors.iss.sublat).split(":")]
        if lat_value[0] < 0:
            lat_value[0] = abs(lat_value[0])
            latitude_ref = "S"
        else:
            latitude_ref = "N"
        latitude = '%d/1,%d/1,%d/10' % (lat_value[0], lat_value[1], lat_value[2]*10)

        camera.exif_tags['GPS.GPSLatitude'] = latitude
        camera.exif_tags['GPS.GPSLongitude'] = longitude
        camera.exif_tags['GPS.GPSLongitudeRef'] = longitude_ref
        camera.exif_tags['GPS.GPSLatitudeRef'] = latitude_ref
        camera.exif_tags['GPS.GPSAltitudeRef'] = "0" 
        camera.exif_tags['GPS.GPSAltitude'] = str( AsPiSensors.iss.elevation)

        latitude_str ='%s%03dd%02dm%02d' % (latitude_ref, lat_value[0], lat_value[1], lat_value[2])
        longitude_str='%s%03dd%02dm%02d' % (longitude_ref, long_value[0], long_value[1], long_value[2])

        return latitude_str ,longitude_str 


class AsPiUserLoop( Thread):
    '''
    Thread that continuously calls the provided callback. Passing as arguments, the results of the 'getdata' function.
    The result of the provided callback is then passed as argument to a 'returndata' function call.
    '''
    def __init__(self, callback , getdata, returndata):
        Thread.__init__(self)
        self.callback = callback
        self.getdata = getdata
        self.returndata = returndata

    def run(self):
        if self.callback is None:
            return
        while True:
            data = self.getdata()
            if data is None:           
                return
            response = self.callback( data )
            self.returndata( response )
            time.sleep(0.5)

class AsPiLogFile:
    '''
    Class that initializes and manages the data log file. 

    A csv data log file, with the specified naming format, is created at the beginning and everytime
    the log file gets bigger than 'logfileMaxBytes' bytes. Each file has a header in the first line
    with the sensors names and the respective units.
    Each data row is written in the csv file as a line with the field values separated by commas with 
    the timestamp in the DATE_FORMAT format as the first field.
    '''
    filePrefix = DEFAULT_LOGFILE_PREFIX

    def __init__(self
            , filePrefix = DEFAULT_LOGFILE_PREFIX
            , logfileMaxBytes = DEFAULT_SIZE_PER_LOGFILE_IN_BYTES
            , sensorList = AsPi.ALL_SENSORS
            , logToStdErr = False ):
        AsPiLogFile.filePrefix = filePrefix 
        self.logToStdErr = logToStdErr
        self.sensorList = sensorList
        self.logfileMaxBytes = logfileMaxBytes
        self.__create_datalogfile()
        
    def __create_datalogfile(self):
        self.currentDatalogFile = AsPiLogFile.generate_fileprefix() + '.' + LOGFILE_EXT
        logzero.logfile( filename=self.currentDatalogFile , disableStderrLogger=not self.logToStdErr)
        self.formatter = Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT)
        logzero.formatter( self.formatter )
        self.__write_header()

    def __write_header(self):
        logzero.formatter( NO_TIMESTAMP_FORMATTER )
        logger.info( self.__generate_header_line() )
        logzero.formatter( self.formatter )

    def __generate_header_line(self):
        commasep_sensor_fields = ' , '.join( [ sensor + " (" + AsPi.UNITS[ sensor ] + ")" for sensor in self.sensorList ] )
        return TIMESTAMP_FIELD + ', ' + commasep_sensor_fields
    
    def log(self, data_array):
        logger.info(",".join( map(str, data_array ) ) )
        if os.path.getsize( self.currentDatalogFile )  > self.logfileMaxBytes:
            self.__create_datalogfile()

    def generate_fileprefix():
        return '{dir}/{prefix}-{timestamp}'.format( dir=CURRENT_DIR , prefix=AsPiLogFile.filePrefix , timestamp=get_timestamp() )    

class AsPiLogger:
    '''
    MAIN CLASS. User facing class that:
        * configures all the options with the user specified values or with the predefined defaults.
        * starts the log timer, to periodically log data from the sensors
        * starts the end timer, to end the program after the specified duration
        * starts the motion detector thread to monitor and register movements event count 
        * if the user callback is specified, it starts the user loop thread to continuously send the collected data 
        to the user provided callback and receive any result to store in the CSV file as a "pseudo" sensor (SENSOR_USERDATA) value 
        * if camera is enabled, starts the camera thread to periodically take pictures with the AstroPi camera
        * Gracefully manages the program finalization phase and abnormal interruption handling (CTRL-C)
    '''
    def __init__(self
            , cameraEnabled = False
            , logPeriodInSecs = DEFAULT_LOG_PERIOD_IN_SECS
            , imgPeriodInSecs = DEFAULT_IMG_PERIOD_IN_SECS
            , filePrefix = DEFAULT_LOGFILE_PREFIX
            , logfileMaxBytes = DEFAULT_SIZE_PER_LOGFILE_IN_BYTES 
            , sensorList = AsPi.ALL_SENSORS
            , durationInSecs = DEFAULT_DURATION_IN_SECS
            , updateCallback = None 
            , logToStdErr = False ):

        AsPiCamera.isCameraEnabled = cameraEnabled
        self.logfile = AsPiLogFile( filePrefix = filePrefix, logfileMaxBytes = logfileMaxBytes, sensorList = sensorList, logToStdErr = logToStdErr )
        self.sensors = AsPiSensors(sense_hat, sensorList)
        self.logPeriodInSecs = logPeriodInSecs if logPeriodInSecs > MIN_LOG_PERIOD_IN_SECS else MIN_LOG_PERIOD_IN_SECS
        self.imgPeriodInSecs = imgPeriodInSecs if imgPeriodInSecs > MIN_IMG_PERIOD_IN_SECS else MIN_IMG_PERIOD_IN_SECS

        self.logtimer = AsPiTimer( self.logPeriodInSecs, self.__log_sensors_reading )
        self.endtimer = AsPiTimer( durationInSecs, AsPi.end )
        self.motiondetector = AsPiMotionDetector()
        if AsPiCamera.isCameraEnabled: 
            self.camera = AsPiCamera( imgPeriodInSecs )
        self.userLoop = AsPiUserLoop( updateCallback , AsPiLogger.getdata , AsPiLogger.setdata )

    def setdata( datareturned ):
        AsPiSensors.userData.put( datareturned )

    def getdata():
        while True:
            lastPictureTaken = AsPiCamera.lastPictureTaken.get() if AsPiCamera.isCameraEnabled else None
            lastAsPiSensorsReading = AsPiSensors.lastAsPiSensorsReading.get()
            if lastPictureTaken is None and lastAsPiSensorsReading is None:
                time.sleep(0.5)
                if AsPi.hasEnded():
                    return None
            else:
                return lastPictureTaken, lastAsPiSensorsReading

    def __log_sensors_reading(self):
        self.logfile.log( self.sensors.read() )
        if not AsPi.hasEnded():
            self.logtimer.reset()

    def __run(self):
        while True:
            if AsPi.hasEnded():
                return

    def start(self): 
        self.userLoop.start()
        self.motiondetector.start()
        self.logtimer.start()
        self.endtimer.start()
        if AsPiCamera.isCameraEnabled:
            self.camera.start()
        try:
            self.__run()
        except KeyboardInterrupt:
            if AsPiCamera.isCameraEnabled: 
                self.camera.stop()
            print("CTRL-C! Exiting…")
        finally:
            # clean up
            AsPi.end()
            self.logtimer.cancel()
            self.endtimer.cancel()
            if AsPiCamera.isCameraEnabled:
                self.camera.join()
            print("Waiting for user callback to finish...")
            self.userLoop.join()
            AsPi.shutdowntimer.cancel()
            print("Program finished.")



TREE_FRAMES = [
[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 64], [18, 255, 0], [18, 255, 0], [155, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [155, 255, 0], [0, 255, 9], [155, 255, 0], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 64], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [155, 255, 0], [0, 255, 64], [0, 255, 9], [155, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [18, 255, 0], [18, 255, 0], [18, 255, 0], [155, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [46, 255, 0], [100, 255, 0], [182, 255, 0], [100, 255, 0], [182, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [46, 255, 0], [155, 255, 0], [0, 255, 9], [182, 255, 0], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [155, 255, 0], [182, 255, 0], [0, 255, 9], [73, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[0, 0, 0], [0, 0, 0], [18, 255, 0], [46, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 64], [18, 255, 0], [18, 255, 0], [155, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [155, 255, 0], [0, 255, 9], [155, 255, 0], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 64], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [155, 255, 0], [0, 255, 64], [0, 255, 9], [155, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[0, 0, 0], [0, 0, 0], [0, 0, 0], [18, 255, 0], [18, 255, 0], [18, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 64], [128, 255, 0], [18, 255, 0], [128, 255, 0], [18, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [155, 255, 0], [128, 255, 0], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [182, 255, 0], [0, 255, 9], [0, 255, 9], [0, 255, 64], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [155, 255, 0], [0, 255, 64], [0, 255, 9], [128, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[0, 0, 0], [0, 0, 0], [0, 0, 0], [46, 255, 0], [46, 255, 0], [18, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 64], [0, 255, 9], [155, 255, 0], [0, 255, 9], [18, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [155, 255, 0], [0, 255, 9], [0, 255, 9], [155, 255, 0], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [155, 255, 0], [155, 255, 0], [0, 255, 64], [46, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [0, 255, 64], [155, 255, 0], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[0, 0, 0], [0, 0, 0], [0, 0, 0], [46, 255, 0], [18, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 64], [0, 255, 9], [155, 255, 0], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [155, 255, 0], [0, 255, 9], [0, 255, 9], [155, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [155, 255, 0], [155, 255, 0], [0, 255, 64], [18, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [0, 255, 64], [155, 255, 0], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[0, 0, 0], [18, 255, 0], [18, 255, 0], [18, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 64], [0, 255, 9], [155, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [155, 255, 0], [0, 255, 9], [0, 255, 9], [155, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [155, 255, 0], [155, 255, 0], [0, 255, 64], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [0, 255, 64], [155, 255, 0], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[0, 0, 0], [0, 0, 0], [18, 255, 0], [18, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [18, 255, 0], [0, 255, 64], [0, 255, 9], [155, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [18, 255, 0], [155, 255, 0], [0, 255, 9], [0, 255, 9], [155, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [18, 255, 0], [0, 255, 9], [155, 255, 0], [155, 255, 0], [0, 255, 64], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [0, 255, 64], [155, 255, 0], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
]

FIRE_TREE_FRAMES = [
[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [0, 255, 36], [0, 255, 36], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 36], [0, 255, 9], [0, 255, 36], [46, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [0, 255, 36], [0, 255, 36], [46, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [0, 255, 36], [0, 255, 36], [0, 255, 36], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 137, 0], [255, 137, 0], [0, 0, 0], [255, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 109, 0], [255, 0, 0], [255, 137, 0], [255, 0, 0], [255, 55, 0], [255, 0, 0], [255, 0, 0], [255, 109, 0], [255, 246, 0], [255, 109, 0], [255, 0, 0], [255, 109, 0], [255, 246, 0], [255, 109, 0]],
[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [0, 255, 9], [0, 255, 9], [0, 255, 9], [255, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [0, 255, 9], [255, 0, 0], [255, 0, 0], [255, 0, 0], [0, 0, 0], [255, 0, 0], [255, 82, 0], [255, 0, 0], [255, 137, 0], [255, 0, 0], [255, 82, 0], [255, 0, 0], [0, 0, 0], [255, 82, 0], [255, 219, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 219, 0], [255, 82, 0], [255, 0, 0], [255, 219, 0], [255, 219, 0], [255, 219, 0], [255, 82, 0], [255, 219, 0], [255, 219, 0], [255, 219, 0], [255, 82, 0]],
[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [46, 255, 0], [18, 255, 0], [18, 255, 0], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [46, 255, 0], [46, 255, 0], [46, 255, 0], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 0, 0], [46, 255, 0], [18, 255, 0], [0, 0, 0], [255, 0, 0], [0, 0, 0], [255, 0, 0], [255, 55, 0], [255, 55, 0], [255, 0, 0], [46, 255, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 55, 0], [255, 55, 0], [255, 55, 0], [255, 55, 0], [255, 0, 0], [255, 0, 0], [255, 55, 0], [255, 55, 0], [255, 55, 0], [255, 219, 0], [255, 219, 0], [255, 55, 0], [255, 0, 0], [255, 55, 0], [255, 219, 0], [255, 55, 0], [255, 219, 0], [255, 219, 0], [255, 219, 0], [255, 219, 0], [255, 55, 0], [255, 219, 0], [255, 219, 0]],
[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 255, 9], [0, 255, 9], [0, 255, 9], [73, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [0, 255, 9], [0, 255, 9], [73, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 55, 0], [255, 0, 0], [0, 255, 9], [73, 255, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 55, 0], [255, 219, 0], [255, 55, 0], [255, 0, 0], [255, 0, 0], [0, 0, 0], [0, 0, 0], [255, 55, 0], [255, 219, 0], [255, 219, 0], [255, 55, 0], [255, 0, 0], [255, 55, 0], [255, 0, 0], [0, 0, 0], [255, 219, 0], [255, 219, 0], [255, 219, 0], [255, 219, 0], [255, 55, 0], [255, 255, 0], [255, 55, 0], [255, 0, 0], [255, 255, 255], [255, 219, 0], [255, 255, 255], [255, 219, 0], [255, 255, 0], [255, 255, 255], [255, 255, 0], [255, 55, 0]],
[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 82, 0], [255, 0, 0], [0, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 137, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 109, 0], [237, 255, 0], [255, 82, 0], [255, 82, 0], [255, 137, 0], [255, 137, 0], [255, 82, 0], [255, 109, 0], [237, 255, 0], [237, 255, 0], [237, 255, 0], [255, 137, 0], [237, 255, 0], [237, 255, 0], [255, 82, 0], [255, 109, 0], [237, 255, 0], [255, 255, 255], [237, 255, 0], [255, 137, 0], [237, 255, 0], [255, 255, 255], [237, 255, 0], [237, 255, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [237, 255, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
]
IN_SPACE = [
[[255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 255, 255], [0, 0, 0], [0, 0, 0], [255, 255, 0]],
[[255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [255, 0, 0], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0]],
[[255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0], [255, 255, 255], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 255, 255], [0, 0, 0], [255, 255, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 246, 0], [255, 246, 0], [255, 0, 0], [0, 0, 0], [255, 255, 0]],
[[255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0], [255, 0, 0], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0], [255, 0, 0], [255, 255, 0], [255, 255, 0], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 246, 0], [255, 246, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0]],
[[255, 255, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 255, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0], [255, 255, 255], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 255, 255], [0, 0, 0], [255, 255, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [0, 0, 0], [255, 255, 255], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 0, 0], [255, 255, 0], [255, 255, 0], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0], [0, 0, 0], [255, 255, 0], [255, 255, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [255, 255, 0]],
[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
]

ROCKET = [
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [128, 0, 255], [128, 0, 255], [128, 0, 255], [128, 0, 255], [128, 0, 255], [128, 0, 255], [128, 0, 255], [128, 0, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [128, 0, 255], [128, 0, 255], [128, 0, 255], [128, 0, 255], [128, 0, 255], [128, 0, 255], [128, 0, 255], [128, 0, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 36, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 200, 255], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 255, 9], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [255, 246, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 82, 0], [255, 82, 0], [255, 255, 255], [255, 255, 255], [255, 82, 0], [255, 82, 0], [255, 255, 255], [255, 255, 255], [255, 82, 0], [255, 82, 0], [255, 255, 255], [255, 255, 255], [255, 82, 0], [255, 82, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [31, 31, 31], [255, 255, 255], [255, 255, 255], [31, 31, 31], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [31, 31, 31], [31, 31, 31], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
[[255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [0, 9, 255], [0, 9, 255], [255, 255, 255], [255, 255, 255], [0, 9, 255], [0, 9, 255], [255, 255, 255], [255, 255, 255], [0, 9, 255], [0, 9, 255], [255, 255, 255], [255, 255, 255], [0, 9, 255], [0, 9, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [31, 31, 31], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 27], [255, 255, 255], [0, 118, 255], [0, 118, 255], [255, 0, 27], [255, 255, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 27], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 27], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [0, 118, 255], [0, 118, 255], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0]],
[[255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 255, 255], [255, 0, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [31, 31, 31], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 146, 255], [31, 31, 31], [0, 146, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 146, 255], [31, 31, 31], [0, 146, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [31, 31, 31], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [31, 31, 31], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [31, 31, 31], [31, 31, 31], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [31, 31, 31], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [31, 31, 31], [0, 118, 255], [31, 31, 31], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [31, 31, 31], [0, 118, 255], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 0, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 246, 0], [255, 246, 0], [255, 0, 0], [0, 118, 255], [0, 118, 255], [255, 82, 0], [255, 82, 0], [191, 191, 191], [255, 246, 0], [255, 246, 0], [191, 191, 191], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [191, 191, 191], [191, 191, 191], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 219, 0], [255, 219, 0], [255, 0, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 146, 255], [255, 246, 0], [255, 246, 0], [0, 146, 255], [0, 118, 255], [0, 118, 255], [255, 82, 0], [255, 82, 0], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 219, 0], [255, 219, 0], [255, 0, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 219, 0], [255, 219, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255], [0, 118, 255], [255, 82, 0], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 255, 255], [255, 0, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 219, 0], [255, 219, 0], [255, 0, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 219, 0], [255, 219, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255], [0, 118, 255], [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255], [255, 82, 0], [255, 82, 0], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 255, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 219, 0], [255, 219, 0], [255, 0, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 219, 0], [255, 219, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255], [0, 118, 255], [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255], [0, 118, 255], [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255], [0, 118, 255], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [255, 0, 0], [255, 219, 0], [255, 219, 0], [255, 0, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 219, 0], [255, 219, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255], [0, 118, 255], [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255], [0, 118, 255], [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[[0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 219, 0], [255, 219, 0], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255], [0, 118, 255], [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255], [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [0, 118, 255], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0], [255, 82, 0]],
[ [0, 118, 255], [0, 118, 255]  , [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255]  , [0, 118, 255] , [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255] , [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255] , [0, 118, 255], [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255] , [0, 118, 255], [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255] , [0, 118, 255], [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255] , [255, 82, 0] , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0] , [255, 82, 0] , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0] ],
[ [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255] , [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255] , [0, 118, 255], [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255] , [0, 118, 255], [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255] , [0, 118, 255], [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255] , [0, 118, 255], [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255] , [255, 82, 0] , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0] , [255, 82, 0] , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0] ],
[ [0, 118, 255], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [191, 191, 191], [0, 118, 255] , [0, 118, 255], [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255] , [0, 118, 255], [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255] , [0, 118, 255], [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255] , [0, 118, 255], [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255] , [0, 118, 255], [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255]  , [0, 118, 255] , [255, 82, 0] , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0] , [255, 82, 0] , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0]   , [255, 82, 0] ]
]

#The animation class holds a list of frames, fps and the current frame of the animation
#frames is list of frames | frame is a list of size 64 containing pixels | a pixel is a list of size 3 holding the rgb values of a given pixel
class Animation:
    def __init__(self, frames, fps = 1, start_frame  = 0, repeat = False):
        self.frames = frames    
        self.fps = fps
        self.frame = 0          #current animation frame
        self.repeat = repeat
    
    def get_frame(self):
        if self.frame >= self.frame_count():
            if not self.repeat:
                return None
            else:
                self.frame = 0
        self.frame += 1
        return self.frames[self.frame - 1]

    def frame_count(self):
        return len(self.frames)

    def set_frame_index(self, index):
        self.frame = index if index >= 0 else 0
    
    def set_repeat(self, repeat):
        self.repeat = repeat

#The AnimationPlayer is the class responsible for playing an animatio on the sense_hat display
#It starts a thread that will play the animation and if any animation is in the queue it starts playing the last one and clears the queue
class AnimationPlayer:
    def __init__(self, sense_hat):
        self.queue = []
        self.queue_lock = threading.Lock()
        self.thread = threading.Thread(target = self.async_player)
        self.new_animation_ev = threading.Event()
        self.exit_ev = threading.Event()
        self.sense = sense_hat

        self.current_animation = None

    def play_animation(self, animation):
        with self.queue_lock:
            self.queue.append(animation)
            self.new_animation_ev.set()

    def queue_length(self):
        with self.queue_lock:
            return len(self.queue)

    def async_player(self):
        while not self.exit_ev.is_set():
            if self.queue_length() == 0:
                if self.current_animation != None:
                    #play new frame
                    frame = self.current_animation.get_frame()
                    if not frame:
                        self.current_animation = None
                        continue

                    self.sense.set_pixels(frame)
                    time.sleep(1/self.current_animation.fps)

                else:
                    self.new_animation_ev.wait()
            else:
                with self.queue_lock:
                    self.current_animation = self.queue[-1]
                    self.queue.clear()
                    self.current_frame = 0

    def start(self):
        self.thread.start()

    def exit(self):
        self.exit_ev.set()

    def exit_and_wait(self):
        self.exit_ev.set()
        self.thread.join()

sense = SenseHat()
sense.low_light = True
sense.set_rotation(270)

#Animation instantiation
ROCKET_ANIMATION = Animation(ROCKET, fps = 2, repeat = True)
NO_FIRE_ANIMATION = Animation(TREE_FRAMES, fps = 0.5, repeat = True)
FIRE_ANIMATION = Animation(FIRE_TREE_FRAMES, fps = 0.5, repeat = True)

anim_player = AnimationPlayer(sense)
anim_player.play_animation(ROCKET_ANIMATION)
anim_player.start()

class ImageProcessor:
    def __init__(self):
        self.imgcounter = 0
        self.recognizer = WildfireRecognizer()

    def process_data(self, data ):
        self.imgcounter = self.imgcounter + 1
        photo, sensor_readings = data
        if photo is None:
            return None
            
        photo_filename, photo_data = photo 

        fire_prob = self.recognizer.recognize_from_image_data(photo_data.get_bytes(), "jpg")

        if fire_prob > 0.5:
            anim_player.play_animation(FIRE_ANIMATION)
        else:
            anim_player.play_animation(NO_FIRE_ANIMATION)

        return Path( photo_filename ).name + "-" + str(fire_prob)

imgproc = ImageProcessor()

aspilogger = AsPiLogger( 
    cameraEnabled = True
    , logPeriodInSecs=2
    , imgPeriodInSecs=15
    , filePrefix="coderdojolx-wildfire"
    , durationInSecs = 3 * 58 * 60
    , logToStdErr=False
    , updateCallback=imgproc.process_data)

aspilogger.start()

anim_player.exit_and_wait()
