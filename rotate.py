from MultiMsgSync import TwoStageHostSeqSync
import blobconverter
import cv2
import depthai as dai
from tools import *
rgbRr = dai.RotatedRect()
def create_pipeline(stereo):
    pipeline = dai.Pipeline()
    print("Creating Color Camera...")
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640,400)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setPreviewKeepAspectRatio(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    if stereo:
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight = pipeline.create(dai.node.MonoCamera)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        stereo = pipeline.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
        depthWidth = 640
        depthHeight = 400
        stereo.setOutputSize(depthWidth,depthHeight)
        # stereo.setOutputSize(1920, 1080)
        monoLeft.out.link(stereo.left)
        monoRight.out.link(stereo.right)
        print("OAK-D detected, app will display spatial coordiantes")
        face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
        manip_depth = pipeline.create(dai.node.ImageManip)
        rotated_rect = dai.RotatedRect()
        rotated_rect.center.x, rotated_rect.center.y = depthWidth // 2, depthHeight // 2
        rotated_rect.size.width, rotated_rect.size.height = depthHeight, depthWidth
        rotated_rect.angle = 90
        manip_depth.initialConfig.setCropRotatedRect(rotated_rect, False)
        stereo.depth.link(manip_depth.inputImage)
        manip_depth.out.link(face_det_nn.inputDepth)
    else:
        print("OAK-1 detected, app won't display spatial coordiantes")
        face_det_nn = pipeline.create(dai.node.MobileNetDetectionNetwork)
    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setBoundingBoxScaleFactor(0.5)
    face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
    copy_manip = pipeline.create(dai.node.ImageManip)
    rgbRr.center.x, rgbRr.center.y = cam.getPreviewWidth() // 2, cam.getPreviewHeight() // 2
    rgbRr.size.width, rgbRr.size.height = cam.getPreviewHeight(), cam.getPreviewWidth()
    rgbRr.angle = 90
    copy_manip.initialConfig.setCropRotatedRect(rgbRr, False)
    copy_manip.setNumFramesPool(15)
    copy_manip.setMaxOutputFrameSize(3499200)
    cam.preview.link(copy_manip.inputImage)
    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
    copy_manip.out.link(face_det_manip.inputImage)
    face_det_manip.out.link(face_det_nn.input)
    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("color")
    copy_manip.out.link(cam_xout.input)
    # face_det_manip.out.link(cam_xout.input)
    face_det_xout = pipeline.create(dai.node.XLinkOut)
    face_det_xout.setStreamName("detection")
    face_det_nn.out.link(face_det_xout.input)
    image_manip_script = pipeline.create(dai.node.Script)
    face_det_nn.out.link(image_manip_script.inputs['face_det_in'])
    face_det_nn.passthrough.link(image_manip_script.inputs['passthrough'])
    cam.preview.link(image_manip_script.inputs['preview'])
    image_manip_script.setScript("""
    import time
    msgs = dict()
    def add_msg(msg, name, seq = None):
        global msgs
        if seq is None:
            seq = msg.getSequenceNum()
        seq = str(seq)
        # node.warn(f"New msg {name}, seq {seq}")
        # Each seq number has it's own dict of msgs
        if seq not in msgs:
            msgs[seq] = dict()
        msgs[seq][name] = msg
        # To avoid freezing (not necessary for this ObjDet model)
        if 15 < len(msgs):
            node.warn(f"Removing first element! len {len(msgs)}")
            msgs.popitem() # Remove first element
    def get_msgs():
        global msgs
        seq_remove = [] # Arr of sequence numbers to get deleted
        for seq, syncMsgs in msgs.items():
            seq_remove.append(seq) # Will get removed from dict if we find synced msgs pair
            # node.warn(f"Checking sync {seq}")
            # Check if we have both detections and color frame with this sequence number
            if len(syncMsgs) == 2: # 1 frame, 1 detection
                for rm in seq_remove:
                    del msgs[rm]
                # node.warn(f"synced {seq}. Removed older sync values. len {len(msgs)}")
                return syncMsgs # Returned synced msgs
        return None
    def correct_bb(xmin,ymin,xmax,ymax):
        if xmin < 0: xmin = 0.001
        if ymin < 0: ymin = 0.001
        if xmax > 1: xmax = 0.999
        if ymax > 1: ymax = 0.999
        return [xmin,ymin,xmax,ymax]
    while True:
        time.sleep(0.001) # Avoid lazy looping
        preview = node.io['preview'].tryGet()
        if preview is not None:
            add_msg(preview, 'preview')
        face_dets = node.io['face_det_in'].tryGet()
        if face_dets is not None:
            # TODO: in 2.18.0.0 use face_dets.getSequenceNum()
            passthrough = node.io['passthrough'].get()
            seq = passthrough.getSequenceNum()
            add_msg(face_dets, 'dets', seq)
        sync_msgs = get_msgs()
        if sync_msgs is not None:
            img = sync_msgs['preview']
            dets = sync_msgs['dets']
            for i, det in enumerate(dets.detections):
                cfg = ImageManipConfig()
                bb = correct_bb(det.xmin-0.03, det.ymin-0.03, det.xmax+0.03, det.ymax+0.03)
                cfg.setCropRect(*bb)
                # node.warn(f"Sending {i + 1}. det. Seq {seq}. Det {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
                cfg.setResize(60, 60)
                cfg.setKeepAspectRatio(False)
                node.io['manip_cfg'].send(cfg)
                node.io['manip_img'].send(img)
    """)
    recognition_manip = pipeline.create(dai.node.ImageManip)
    recognition_manip.initialConfig.setResize(60, 60)
    recognition_manip.setWaitForConfigInput(True)
    image_manip_script.outputs['manip_cfg'].link(recognition_manip.inputConfig)
    image_manip_script.outputs['manip_img'].link(recognition_manip.inputImage)
    print("Creating recognition Neural Network...")
    recognition_nn = pipeline.create(dai.node.NeuralNetwork)
    recognition_nn.setBlobPath(blobconverter.from_zoo(name="head-pose-estimation-adas-0001", shaves=6))
    recognition_manip.out.link(recognition_nn.input)
    recognition_xout = pipeline.create(dai.node.XLinkOut)
    recognition_xout.setStreamName("recognition")
    xoutDepth = pipeline.create(dai.node.XLinkOut)
    xoutDepth.setStreamName("depth")
    face_det_nn.passthroughDepth.link(xoutDepth.input)
    recognition_nn.out.link(recognition_xout.input)
    return pipeline
with dai.Device() as device:
    stereo = 1 < len(device.getConnectedCameras())
    device.startPipeline(create_pipeline(stereo))
    sync = TwoStageHostSeqSync()
    queues = {}
    for name in ["color", "detection", "recognition"]:
        queues[name] = device.getOutputQueue(name)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    while True:
        depth = depthQueue.get()
        for name, q in queues.items():
            # Add all msgs (color frames, object detections and recognitions) to the Sync class.
            if q.has():
                sync.add_msg(q.get(), name)
        msgs = sync.get_msgs()
        depthFrame = depth.getFrame()  # depthFrame values are in millimeters
        depth_downscaled = depthFrame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0  # Set a default minimum depth value when all elements are zero
        else:
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_TURBO)
        if msgs is not None:
            frame = msgs["color"].getCvFrame()
            detections = msgs["detection"].detections
            recognitions = msgs["recognition"]
            for i, detection in enumerate(detections):
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                roiData = detection.boundingBoxMapping
                roi = roiData.roi

                depthroi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                depthtopLeft = depthroi.topLeft()
                depthbottomRight = depthroi.bottomRight()
                depthxmin = int(depthtopLeft.x)
                depthymin = int(depthtopLeft.y)
                depthxmax = int(depthbottomRight.x)
                depthymax = int(depthbottomRight.y)
                cv2.rectangle(depthFrameColor, (depthxmin, depthymin), (depthxmax, depthymax),  (255, 255, 255), 1)

                roi = roi.denormalize(frame.shape[1], frame.shape[0])   # Normalize bounding box

                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmin = int(topLeft.x)
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)
                ymax = int(bottomRight.y)
                # Decoding of recognition results
                rec = recognitions[i]
                yaw = rec.getLayerFp16('angle_y_fc')[0]
                pitch = rec.getLayerFp16('angle_p_fc')[0]
                roll = rec.getLayerFp16('angle_r_fc')[0]
                decode_pose(yaw, pitch, roll, bbox, frame)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 0), 2)
                y = (bbox[1] + bbox[3]) // 2
                if stereo:
                    coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z/1000)
                    cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 8)
                    cv2.putText(frame, coords, (bbox[0], y + 60), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)
            height, width, channels = frame.shape
            print(f"Frame dimensions: Width={width}, Height={height}, Channels={channels}")
            cv2.imshow("Camera", frame)
            cv2.imshow("depth", depthFrameColor)
        if cv2.waitKey(1) == ord('q'):
            break