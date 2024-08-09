from MultiMsgSync import TwoStageHostSeqSync
import blobconverter
import depthai as dai
from tools import *
def create_pipeline():
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setPreviewSize(640,400)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setInterleaved(False)
    cam.setPreviewKeepAspectRatio(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    monoLeft = pipeline.create(dai.node.MonoCamera)
    monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
    monoRight = pipeline.create(dai.node.MonoCamera)
    monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setOutputSize(640,400)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)
    face_det_nn = pipeline.create(dai.node.MobileNetSpatialDetectionNetwork)
    manip_depth = pipeline.create(dai.node.ImageManip)
    rotated_rect = dai.RotatedRect()
    rotated_rect.center.x, rotated_rect.center.y = 640 // 2, 400 // 2
    rotated_rect.size.width, rotated_rect.size.height = 400, 640
    rotated_rect.angle = 90
    manip_depth.initialConfig.setCropRotatedRect(rotated_rect, False)
    stereo.depth.link(manip_depth.inputImage)
    manip_depth.out.link(face_det_nn.inputDepth)
    face_det_nn.setConfidenceThreshold(0.5)
    face_det_nn.setBoundingBoxScaleFactor(0.5)
    face_det_nn.setBlobPath(blobconverter.from_zoo(name="face-detection-retail-0004", shaves=6))
    copy_manip = pipeline.create(dai.node.ImageManip)
    rgbRr = dai.RotatedRect()
    rgbRr.center.x, rgbRr.center.y = cam.getPreviewWidth() // 2, cam.getPreviewHeight() // 2
    rgbRr.size.width, rgbRr.size.height = cam.getPreviewHeight(), cam.getPreviewWidth()
    rgbRr.angle = 90
    copy_manip.initialConfig.setCropRotatedRect(rgbRr, False)
    copy_manip.setNumFramesPool(15)
    cam.preview.link(copy_manip.inputImage)
    face_det_manip = pipeline.create(dai.node.ImageManip)
    face_det_manip.initialConfig.setResize(300, 300)
    face_det_manip.initialConfig.setFrameType(dai.RawImgFrame.Type.RGB888p)
    copy_manip.out.link(face_det_manip.inputImage)
    face_det_manip.out.link(face_det_nn.input)
    cam_xout = pipeline.create(dai.node.XLinkOut)
    cam_xout.setStreamName("color")
    copy_manip.out.link(cam_xout.input)
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
        if seq not in msgs:
            msgs[seq] = dict()
        msgs[seq][name] = msg
        if 15 < len(msgs):
            node.warn(f"Removing first element! len {len(msgs)}")
            msgs.popitem() 
    def get_msgs():
        global msgs
        seq_remove = []
        for seq, syncMsgs in msgs.items():
            seq_remove.append(seq)
            if len(syncMsgs) == 2:
                for rm in seq_remove:
                    del msgs[rm]
                return syncMsgs 
        return None
    def correct_bb(xmin,ymin,xmax,ymax):
        if xmin < 0: xmin = 0.001
        if ymin < 0: ymin = 0.001
        if xmax > 1: xmax = 0.999
        if ymax > 1: ymax = 0.999
        return [xmin,ymin,xmax,ymax]
    while True:
        time.sleep(0.001) 
        preview = node.io['preview'].tryGet()
        if preview is not None:
            add_msg(preview, 'preview')
        face_dets = node.io['face_det_in'].tryGet()
        if face_dets is not None:
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
    device.startPipeline(create_pipeline())
    sync = TwoStageHostSeqSync()
    queues = {}
    for name in ["color", "detection", "recognition"]:
        queues[name] = device.getOutputQueue(name)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
    while True:
        depth = depthQueue.get()
        for name, q in queues.items():
            if q.has():
                sync.add_msg(q.get(), name)
        msgs = sync.get_msgs()
        depthFrame = depth.getFrame()
        depth_downscaled = depthFrame[::4]
        if np.all(depth_downscaled == 0):
            min_depth = 0
        else:
            min_depth = np.percentile(depth_downscaled[depth_downscaled != 0], 1)
        max_depth = np.percentile(depth_downscaled, 99)
        depthFrameColor = np.interp(depthFrame, (min_depth, max_depth), (0, 255)).astype(np.uint8)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_TURBO)
        if msgs is not None:
            frame = msgs["color"].getCvFrame()
            detections = msgs["detection"].detections
            for i, detection in enumerate(detections):
                bbox = frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                roiData = detection.boundingBoxMapping
                roi = roiData.roi
                roi = roi.denormalize(frame.shape[1], frame.shape[0])
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (10, 245, 10), 2)
                cv2.rectangle(frame, (int(topLeft.x), int(topLeft.y)), (int(bottomRight.x), int(bottomRight.y)), (0, 0, 0), 2)
                cv2.rectangle(depthFrameColor, (int(topLeft.x), int(topLeft.y)), (int(bottomRight.x), int(bottomRight.y)), (0, 0, 0),2)
            cv2.imshow("Camera", frame)
            cv2.imshow("depth", depthFrameColor)
        if cv2.waitKey(1) == ord('q'):
            break
