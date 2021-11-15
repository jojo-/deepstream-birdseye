# Bird's eye view with DeepStream

## Introduction

This project shows how to generate bird's eye view visualisations of an intersection and its traffic using DeepStream 6.0.

The application performs **in real-time** the following steps:

1. Video stream capture and decoding
2. Primary inference engine detects the vehicles in the incoming frames
3. The detected objects are send to a secondary inference engines, a vehicle type classifier
4. The objects are given an unique id thanks to a tracker
5. The stream and coordinates of the detected objects are then warped to enable a bird's eye view of the intersection
6. Both the original video and the bird's eye view with the

## Method

### 1. Converting the original view to a bird's eye view

Let's assume we have this camera view:

![cam_view](/doc/camera_view.png)

To convert that to a bird's eye (or top-down) view, we need to find the proper perpsective transfor (or homography).
This can be done by either using the camera intrinsic (eg focal length) and extrinsinc paramterst (eg roll, pitch, elevation) or finding 4 corresponding landmarks on the camera view and a top-down view.


In our case, the parameters of the camera are unknow. On the other hand, it is possible to use [Google Map](https://www.google.be/maps/@39.4703481,-0.3849947,77m/data=!3m1!1e3) to obtain a top-down image of the intersection being monitored by the camera. This top-down view is illustrated below.

![top_down_view](/doc/top_down.png)

We can then determine the pixel coordinates of 4 **co-planar** landmarks visible both in the camera view and the top-down view as illustrated below.

![landmark](/doc/landmarks.png)

We can then pass the pixel coordinate of 4 landmarks in the camera view and in the top-down view to the OpenCV function `getPerspectiveTransform` to compute the perspective transformation matrix, ie the matrix allowing to convert pixel coordinate in the camera view to the to- down view

The output of the perspective transform can be see in right panel of the figure below. One can see that the blue dot at coordinate [1026, 408]  below a traffic sign at the corner of the grass area has been projected on the right location in the bird's eye view.

![perspective_transform](/doc/perspective_transform_res.png)

The functions to perform those steps are available in the `perspective.py` file.

### 2. Create a clean background (optional)

Instead of using the bird's eye view from the camera view, which can be a bit messy and has distorded objects, we can use a map

![map_bev](/input/background.png)

### 3. Detecting and tracking vehicles

The detection of the vehicles is done using the original camera view. The object detection is performed using the `TrafficCamNet` model available from the [NVIDIA NGC repository](https://catalog.ngc.nvidia.com/orgs/nvidia/models/tlt_trafficcamnet). TrafficCamNet is based on the NVIDIA DetecNet_v2 detector and rely on a Resnet18 backbone for the feature extractor. The model has been pruned for computational efficiency (unpruned model is 44.32 MB while the pruned version is only 5.20 MB). The application uses TensorRT as the inference engine.

To associate unique IDs to the detections while they remain in the field of view of the camera, we can use a tracker. The [NvDCF](https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_plugin_gst-nvtracker.html) algoritm employed in this application  is a visual tracker that is based on the discriminative correlation filter (DCF) for learning a target-specific correlation filter and for localizing the same target in the next frames using the learned correlation filter.

For each tracked target, the NvDCF tracker defines a search region around its predicted location in the next frame large enough for the same target to be detected in the search region. The association of target IDs across frames for robust tracking entails visual appearance-based similarity matching, for which the visual appearance features are extracted at each candidate location. 

An example of the ouput is shown below. The id next to the type of the detection is provided by the tracking algorithm (see the next sub-section).

![detection](/doc/detection.png)

Please note that TrafficCamNet has not been retrained for this project, so the accuracy is not optimal for this context, but using the TAO framework, this is something that can be easily done using [NVIDIA TAO](https://docs.nvidia.com/metropolis/TLT/tlt-user-guide/text/object_detection/detectnet_v2.html).

## Putting everything together: Deepstream application overview

Assume that the homography matrix is available, the application continuously performs the following steps
1. Capture a frame from the camera view
2. Perform detection on the frame
3. Pass the detection to the tracker
4. Convert the detection coordinate. This in done in parrallel in order to not slow down the detection and tracking steps.

The high level architecture of the application is illustrated in the figure below.

![ds](/doc/deepstream.png)

### Output

The ouput of the application can be seen below:

1. Detection and tracking using the camera view

![det_cam_view](/doc/out_cam_view.png)

2. bird's eye view of the detection

![bev_det](/doc/bev_det.png)

3. Localisation of the detection on the map (red markers)

![bev_map](/doc/bev_map_det.png)

## Usage

```
python3 ds-bev.py <config_file>
```

A sample config file is provided in `config_app.txt`

### Configuration files

See the `config_app.txt` file for the different configuration options of the application. You can configure:
- the outputs: video files, display;
- the source video feed;
- the FPS indicator;
- the background image for the bird's eye view projection.

The different models hava their own configuration file in the `config` directory:
- `config_infer_primary_trafficamnet.txt` for the vehicle detector;
- `config_infer_secondary_vehicletypenet.txt` for the vehicle type classifier;
- `config_tracker.txt` for the tracker.

If you encounter an error when trying to run the application, or if DeepStream keeps rebuiling the inference
engine files you can modify the paths in the configuration file (ie use absolute paths instead of relative ones).

## Installation steps

Those steps are for an x86-64 platform running Ubuntu 18.04 LTS, but should similar for the Jetson platform.

### DeepStream 6.0 and its Python 3 bindings (pyds):

More information here: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html

- Install dependenceies
```
sudo apt install \
libssl1.0.0 \
libgstreamer1.0-0 \
gstreamer1.0-tools \
gstreamer1.0-plugins-good \
gstreamer1.0-plugins-bad \
gstreamer1.0-plugins-ugly \
gstreamer1.0-libav \
libgstrtspserver-1.0-0 \
libjansson4=2.11-1 \
python3-pip
```

- Install DeepStream:

Download the DeepStream 6.0 Jetson Debian package deepstream-6.x_arm64.deb. Then enter the command:
```
sudo apt-get install ./deepstream-6.x_arm64.deb
```

- Enable MetaData access:

```
cd /opt/nvidia/deepstream/deepstream/lib
python3 setup.py install
```

### Python 3 packages

In addition to the DeepStream dependencies, you need to install the `cv2` package for OpenCV. The package `matplotlib` is needed to test the functions in `perspective.py`, but is not required to run the application.

## List of potential improvements

This project has been coded in a single afternoon, thus there are a lot of potential improvement. For instance:

- If the intrinsinc parameters of the CCTV camera are know, this can be used to create a better bird's eye view projection, as well as removing the distortion of the input stream
- The video quality of the stream should be improved for optimal results (the video is pixelated, maybe low bitrate, or old encoding standard).
- While the tracker is active (the detections all have a unique id while they stay in the field of view of the camera), the trajectories are not recorded so far
- A heat map of the detection can also be a great addition to the visualisation, this will allow the bottleneck in the
  intersections
- The object detector at the core of the DeepStream application is based on a Detectnet v2 architecture relying on a ResNet-18
  backbone. While this is sufficien to build a quick proof of concept, the model should be either retrained or replaced (eg with
  a YoloV4 model) to improvde the detection accuracy and remove the false positives (see the billboard).
- The application can easily be containerized and deployed at the edge in Jetson platforms.
- Using INT8 precision instead of FP32/FP16.
- Speed estimation can be added.

### A quick note on the estimation of the camera's parameters

[CameraTransform](https://github.com/rgerum/cameratransform) is Python package that can be used to estimate camera parameters and apply them to project points from the camera space to the world space and back.

Using the package, and the GPS coordinate of pixels in the camera view derived from Google Maps, it was possible to estimate:
- the elevation of the camera: 30.2m;
- the focal lenght: 3.01mm;
- the tilt of the camera: 72.0 degrees; 
- and the heading of the camera: 121.8 degrees.

The results of the estimation process is illustrated in the figure below. While this approach is promising, in practice, using the estimated parameters to create the homography matrix did not provide a better accuracy for the bird's eye view.

![cam_param_estimation](/doc/camera_parameter_estimation.png)

## Contact

Johan Barthelemy - johan.barthelemy@gmail.com
