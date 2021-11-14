#!/usr/bin/python3

################################################################################
# Demo - Creating Bird Eye View of an Intersection
#
# Version: 14 Nov 2021
# Author : Johan Barthelemy - johan.barthelemy@gmail.com
#
# License: MIT
# Copyright (c) 2021 Johan Barthelemy
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
################################################################################ 

################################################################################
# This work relies on some functions initially written by NVIDIA and modified
# by the author:
# - cb_newpad
# - decodebin_child_added
# - create_source_bin
#
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import sys
import configparser
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst
from common.bus_call import bus_call
from common.FPS import GETFPS
import pyds
import numpy as np
import cv2
import os
import threading
import perspective
from multiprocessing import  Process, Pipe

# To exchange frame with a separate process
recv_cv2_frame, send_cv2_frame = Pipe()

# Primary inference engine
PGIE_CLASS_ID_VEHICLE = 0
PGIE_CLASS_ID_PERSON  = 1
PGIE_CLASS_ID_BICYCLE = 2

# Display output
DISPLAY_OUT = 1

# FPS
FPS_OUT = 0
fps_stream = GETFPS(0)

# Perspective transformation matrix
TRANS_MATRIX = perspective.ipm()

# Backgound for bird eye view
BACKGROUND_IMAGE = cv2.imread('input/background.png')
USE_MAP          = False

# Enabling/disable recording of bird eye view
SAVE_BEV = False
BEV_VIDEO_WRITER = None

# Removing outliers
# list of areas defined by (x_min,y_min, x_max, y_max) where not to show markers
CLEAN_OUTLIERS = [
    (733,800,994,1072),
    (252,781,444,939)
]

# Display the transformed frame to the screen
def show_video(recv_frame):

    
    if SAVE_BEV:
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        outwriter = cv2.VideoWriter('out/bev.avi', fourcc, 30, (1920,1080))

    while True:
        frame, list_cent = recv_frame.recv()        

        if frame is not None:

            global BEV_VIDEO_WRITER

            if USE_MAP:
                n_frame = BACKGROUND_IMAGE.copy()
            else:
                n_frame = warp_image(frame)                    
            
            if len(list_cent) > 0:
                list_cent = perspective.convert_set_cooordinates(list_cent, TRANS_MATRIX)
            
                for i in list_cent:                    

                    keep = True
                    x, y = (int(i[0]), int(i[1]))
                    for o in CLEAN_OUTLIERS:
                        if (o[0] < x < o[2]) and (o[1] < y < o[3]):
                            keep = False
                    
                    if keep:
                        cv2.drawMarker(n_frame, (int(i[0]), int(i[1])),(255,0,0), markerType=cv2.MARKER_STAR, markerSize=20, thickness=3, line_type=cv2.LINE_AA)                    


            n_frame = cv2.cvtColor(n_frame, cv2.COLOR_RGBA2BGR)                           

            if SAVE_BEV:                             
                outwriter.write(n_frame)

            if DISPLAY_OUT:
                cv2.imshow("bird-eye", n_frame)
                cv2.waitKey(1)

# Applying transformation matrix to an image
def warp_image(image, size_img=(1920, 1080), M=TRANS_MATRIX):
    
    warped = cv2.warpPerspective(image, M, size_img, flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped

# Probe to get the image buffer
def osd_sink_pad_buffer_probe(pad,info,u_data):
    
    frame_number = 0
    #Intiallizing object counter with 0    
    obj_counter = {
        PGIE_CLASS_ID_VEHICLE: 0,
        PGIE_CLASS_ID_PERSON: 0,
        PGIE_CLASS_ID_BICYCLE: 0,            
    }    

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))    

    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
           frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
           
        except StopIteration:
            break
            
        frame_number = frame_meta.frame_num        
        l_obj        = frame_meta.obj_meta_list
        
        list_centroids = []

        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)                                
            except StopIteration:
                break
                                                                                                                                         
            obj_counter[obj_meta.class_id] += 1
                        
            # Current object is a vehicle
            if (obj_meta.class_id == PGIE_CLASS_ID_VEHICLE):                
                                                        
                # Getting the labels from the classifier
                # ... can be used later for giving different colors to different categories
                
                veh_type = "car"
                l_obj_meta_cl = obj_meta.classifier_meta_list
                while l_obj_meta_cl is not None:

                    try:
                        cls_meta = pyds.NvDsClassifierMeta.cast(l_obj_meta_cl.data)
                    except StopIteration:
                        break
                                                    
                    l_labels_meta = cls_meta.label_info_list
                    while l_labels_meta is not None:
                        try:
                            lbl_meta = pyds.NvDsLabelInfo.cast(l_labels_meta.data)
                        except StopIteration:
                            break
                            
                        veh_type = lbl_meta.result_label                        
                        
                        try:
                            l_labels_meta = l_labels_meta.next
                        except StopIteration:
                            break
                                                
                    try: 
                        l_obj_meta_cl = l_obj_meta_cl.next
                    except StopIteration:
                        break

                # retrieve coordinates
                rect_params = obj_meta.rect_params                
                top    = int(rect_params.top)
                left   = int(rect_params.left)
                width  = int(rect_params.width)
                height = int(rect_params.height)
                obj_centroid = [left + width * 0.5, top + height * 0.5]

                list_centroids.append(obj_centroid)

            try: 
                l_obj=l_obj.next
            except StopIteration:
                break
                                            
        # Copying and sending the frame to separate process
        n_frame = pyds.get_nvds_buf_surface(hash(gst_buffer), frame_meta.batch_id)                
        send_cv2_frame.send((n_frame, list_centroids))                                                                                                                                         
               
        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta = pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]

        # Setting display text to be shown on screen
        py_nvosd_text_params.display_text = "Frame Number={} Vehicle_count={}".format(frame_number, obj_counter[PGIE_CLASS_ID_VEHICLE])

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10        
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1        
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)        
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        if FPS_OUT == 1:
            fps_stream.get_fps()

        try:
            l_frame=l_frame.next
        except StopIteration:
            break
			
    return Gst.PadProbeReturn.OK

# Callback when a new pad is created by GStreamer for the decodebin
def cb_newpad(decodebin, decoder_src_pad, data):
    print("In cb_newpad\n")
    caps=decoder_src_pad.get_current_caps()
    gststruct=caps.get_structure(0)
    gstname=gststruct.get_name()
    source_bin=data
    features=caps.get_features(0)

    # Need to check if the pad created by the decodebin is for video and not audio
    if(gstname.find("video")!=-1):
        # Link the decodebin pad only if decodebin has picked nvidia decoder plugin nvdec_*.
        if features.contains("memory:NVMM"):
            # Get the source bin ghost pad
            bin_ghost_pad=source_bin.get_static_pad("src")
            if not bin_ghost_pad.set_target(decoder_src_pad):
                sys.stderr.write("Failed to link decoder src pad to source bin ghost pad\n")
        else:
            sys.stderr.write(" Error: Decodebin did not pick an NVIDIA decoder plugin.\n")

# Callback when a child element is added to decodebin
def decodebin_child_added(child_proxy,Object,name,user_data):
    print("Decodebin child added:", name, "\n")
    if(name.find("decodebin") != -1):
        Object.connect("child-added",decodebin_child_added,user_data)   

# Creating the gstreamer container to decode a source        
def create_source_bin(uri):
    print("Creating source bin")

    # Create a source GstBin to abstract this bin's content from the rest of the
    # pipeline
    bin_name="source-bin"
    nbin=Gst.Bin.new(bin_name)
    if not nbin:
        sys.stderr.write(" Unable to create source bin \n")

    # Source element for reading from the uri.
    # We will use decodebin and let it figure out the container format of the
    # stream and the codec and plug the appropriate demux and decode plugins.
    uri_decode_bin=Gst.ElementFactory.make("uridecodebin", "uri-decode-bin")
    if not uri_decode_bin:
        sys.stderr.write(" Unable to create uri decode bin \n")
    # We set the input uri to the source element
    uri_decode_bin.set_property("uri",uri)
    # Connect to the "pad-added" signal of the decodebin which generates a
    # callback once a new pad for raw data has beed created by the decodebin
    uri_decode_bin.connect("pad-added",cb_newpad,nbin)
    uri_decode_bin.connect("child-added",decodebin_child_added,nbin)

    # We need to create a ghost pad for the source bin which will act as a proxy
    # for the video decoder src pad. The ghost pad will not have a target right
    # now. Once the decode bin creates the video decoder and generates the
    # cb_newpad callback, we will set the ghost pad target to the video decoder
    # src pad.
    Gst.Bin.add(nbin,uri_decode_bin)
    bin_pad=nbin.add_pad(Gst.GhostPad.new_no_target("src",Gst.PadDirection.SRC))
    if not bin_pad:
        sys.stderr.write(" Failed to add ghost pad in source bin \n")
        return None
    return nbin        

def main(args):

    # Check input arguments
    if len(args) != 2:
        sys.stderr.write("usage: %s <config file>\n" % args[0])
        sys.exit(1)

    # Reading configuration file
    config_app = configparser.ConfigParser()
    config_app.read(args[1])
    config_app.sections()
    
    for key in config_app['source']:
        if key == 'uri':
            URI_INPUT = config_app.get('source', key)

    for key in config_app['output']:
        if key == 'enable-display':
            global DISPLAY_OUT
            DISPLAY_OUT = config_app.getint('output', key)
        if key == 'enable-file-out':
            FILE_OUT = config_app.getint('output', key)
        if key == 'enable-fps':
            global FPS_OUT
            FPS_OUT = config_app.getint('output', key)
        if key == 'use-map-background':
            global USE_MAP
            USE_MAP = config_app.getboolean('output', key)
        if key == 'enable-bev-out':
            global BEV_VIDEO_WRITER
            global SAVE_BEV
            SAVE_BEV = config_app.getboolean('output', key)
            #if SAVE_BEV:                
                #fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                #BEV_VIDEO_WRITER = cv2.VideoWriter('out/bev.avi', fourcc, 30, (1920,1080))

            


    # Starting separate process for bird eye view visualisation
    p2 = Process(target=show_video, args=(recv_cv2_frame,))
    p2.start()
            
    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create GStreamer Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    print("Creating Source \n ")
    source = create_source_bin(URI_INPUT)
    if not source:
        sys.stderr.write("Unable to create source bin \n")       
        
    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")
    
    # Use nvinfer to run inferencing on camera's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    # Add a tracker
    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    if not tracker:
        sys.stderr.write(" Unable to create tracker \n")
        
    # Add a secondary inference engine (vehicle type)
    sgie = Gst.ElementFactory.make("nvinfer", "secondary2-nvinference-engine")
    if not sgie:
        sys.stderr.write(" Unable to make sgie2 \n")

    # Use convertor to convert from NV12 to RGBA as required by nvosd
    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")
   
    # Sink
    print("Creating sink(s) \n")

    # For splitting the pipeline if sinks in parallel
    tee=None
         
    # No sinks, fake output       
    if DISPLAY_OUT + FILE_OUT == 0:
        print(" fakesink selected")
        sink_fake = Gst.ElementFactory.make("fakesink", "fake-sink")    
        if not sink_fake:
            sys.stderr.write(" Unable to create fake display sink \n")
        sink_fake.set_property('sync', False)
    
    else:                    
        tee=Gst.ElementFactory.make("tee", "nvsink-tee")
        if not tee:
            sys.stderr.write(" Unable to create tee \n")
            
    if DISPLAY_OUT == 1:
        print(" display sink selected")
        
        queue_disp=Gst.ElementFactory.make("queue", "nvtee-q-disp")
        if not queue_disp:
            sys.stderr.write(" Unable to create queue for display \n")                    
        
        sink_disp = Gst.ElementFactory.make("nveglglessink", "nvvideo-renderer")    
        if not sink_disp:
            sys.stderr.write(" Unable to create display sink \n")    
        # Set sync = false to avoid late frame drops at the display-sink
        sink_disp.set_property('sync', False)            
               
    if FILE_OUT == 1:
    
        print("Creating file sink")
              
        queue_file=Gst.ElementFactory.make("queue", "nvtee-q-file")
        if not queue_file:
            sys.stderr.write(" Unable to create queue for file \n")
        
        nvvidconv_postosd_f = Gst.ElementFactory.make("nvvideoconvert", "convertor_postosd_f")
        if not nvvidconv_postosd_f:
            sys.stderr.write(" Unable to create nvvidconv_postosd_f \n")
                        
        encoder_f = Gst.ElementFactory.make("nvv4l2h265enc", "encoder_f")        
        if not encoder_f:
            sys.stderr.write(" Unable to create H265 encoder for file")
                         
        parser_f = Gst.ElementFactory.make("h265parse", "h265parser_f")        
        if not parser_f:
            sys.stderr.write(" Unable to create H265 parser for file")
            
        qtmux = Gst.ElementFactory.make("qtmux", "muxer")
        if not encoder_f:
            sys.stderr.write(" Unable to create muxer")   
            
        sink_file = Gst.ElementFactory.make("filesink", "filesink")
        sink_file.set_property('location', 'out/out.h265')                                                
    
    # Set properties of streamux
    streammux.set_property('width', 1920)
    streammux.set_property('height', 1080)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 4000000)    
      
    # Use CUDA unified memory in the pipeline so frames can be easily accessed on CPU
    mem_type = int(pyds.NVBUF_MEM_CUDA_UNIFIED)
    streammux.set_property("nvbuf-memory-type", mem_type)
    nvvidconv.set_property("nvbuf-memory-type", mem_type)          

    # Set properties of inference engines
    pgie.set_property('config-file-path', "config/config_infer_primary_trafficcamnet.txt")
    sgie.set_property('config-file-path', "config/config_infer_secondary_vehicletypenet.txt")
        
    # Set properties of tracker
    config_tracker = configparser.ConfigParser()
    config_tracker.read('config/config_tracker.txt')
    config_tracker.sections()
    
    for key in config_tracker['tracker']:
        if key == 'tracker-width' :
            tracker_width = config_tracker.getint('tracker', key)
            tracker.set_property('tracker-width', tracker_width)
        if key == 'tracker-height' :
            tracker_height = config_tracker.getint('tracker', key)
            tracker.set_property('tracker-height', tracker_height)
        if key == 'gpu-id' :
            tracker_gpu_id = config_tracker.getint('tracker', key)
            tracker.set_property('gpu_id', tracker_gpu_id)
        if key == 'll-lib-file' :
            tracker_ll_lib_file = config_tracker.get('tracker', key)
            tracker.set_property('ll-lib-file', tracker_ll_lib_file)
        if key == 'll-config-file' :
            tracker_ll_config_file = config_tracker.get('tracker', key)
            tracker.set_property('ll-config-file', tracker_ll_config_file)
        if key == 'enable-batch-process' :
            tracker_enable_batch_process = config_tracker.getint('tracker', key)
            tracker.set_property('enable_batch_process', tracker_enable_batch_process)
        
    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(tracker)
    pipeline.add(sgie)    
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
      
    if DISPLAY_OUT + FILE_OUT == 0:
        pipeline.add(sink_fake)    
    else:
        pipeline.add(tee)
        if DISPLAY_OUT == 1:
            pipeline.add(queue_disp)       
            pipeline.add(sink_disp)                  
        if FILE_OUT == 1:
            pipeline.add(queue_file)
            pipeline.add(nvvidconv_postosd_f)
            pipeline.add(encoder_f)
            pipeline.add(parser_f)
            pipeline.add(qtmux)
            pipeline.add(sink_file)         

    # Linking the elements together
    # source -> mux -> nvinfer (pri) -> tracker -> nvinfer (sec) -> nvinfer (sec) -> nvvideoconvert -> nvosd -> sink
    print("Linking elements in the Pipeline \n")
    
    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = None
        
    srcpad = source.get_static_pad("src")   
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of caps_vidconvsrc \n")
    srcpad.link(sinkpad)

    streammux.link(pgie)   
    pgie.link(tracker)
    tracker.link(sgie)
    sgie.link(nvvidconv)    
    nvvidconv.link(nvosd)
    
    if DISPLAY_OUT + FILE_OUT == 0:
        nvosd.link(sink_fake)
    
    else:
        nvosd.link(tee)
        
        if DISPLAY_OUT == 1:
            sink_pad_q_disp = queue_disp.get_static_pad("sink")
            tee_disp_pad = tee.get_request_pad('src_%u')
            tee_disp_pad.link(sink_pad_q_disp)
            
            if not tee_disp_pad:
                sys.stderr.write("Unable to get requested tee src pads for display\n")
       
            queue_disp.link(sink_disp)
                                    
        if FILE_OUT == 1:
            sink_pad_q_file  = queue_file.get_static_pad("sink")
            tee_file_pad = tee.get_request_pad('src_%u')
            tee_file_pad.link(sink_pad_q_file)
            
            if not tee_file_pad:
                sys.stderr.write("Unable to get requested tee src pads for file\n")
            
            queue_file.link(nvvidconv_postosd_f)
            nvvidconv_postosd_f.link(encoder_f)
            encoder_f.link(parser_f)
            parser_f.link(qtmux)
            qtmux.link(sink_file)       
        

    # Create an event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)
        
    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")

    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # Start play back and listen to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    except:
        pass           
                   
    # Cleanup
    pipeline.set_state(Gst.State.NULL)
    p2.terminate()
    p2.join()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
