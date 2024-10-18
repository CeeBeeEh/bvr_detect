# bvr_detect

<div style="text-align: center;">
  <img alt="BVR Chirp Logo" src="logo.png" width="380" />
</div>


BVR Detect is an object detection library to be used as part of BVR (Bvr Video Recorder).

### THIS PROJECT IS IN ALPHA. It should work, but don't expect too much yet.

# Usage

This should be imported as a local crate in another project, such as [BvrWebPup](https://github.com/CeeBeeEh/bvr_web_pup). 

You would have something like this in the other project: 

`bvr_detect = { version = "0.2.0", path = "../bvr_detect" }`

# Functionality

This is purely a library to receive an input image, run inference, process the detections, and return the results.

# Acknowledgements 

- https://github.com/pykeio/ort
- https://github.com/AlexeyAB/darknet
- https://github.com/WongKinYiu/YOLO
- https://github.com/WongKinYiu/yolov9
- https://github.com/WongKinYiu/yolov7
- https://github.com/ultralytics/ultralytics
- https://github.com/jamjamjon/usls
