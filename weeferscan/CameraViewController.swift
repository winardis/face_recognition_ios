//
//  Copyright (c) 2018 Google Inc.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//

import AVFoundation
import CoreVideo
import MLImage
import MLKit


@objc(CameraViewController)
class CameraViewController: UIViewController {
    private let detectors: [Detector] = [
        .onDeviceFace,
        .onDeviceText,
        .onDeviceBarcode,
        .onDeviceImageLabel,
        .onDeviceImageLabelsCustom,
        .onDeviceObjectProminentNoClassifier,
        .onDeviceObjectProminentWithClassifier,
        .onDeviceObjectMultipleNoClassifier,
        .onDeviceObjectMultipleWithClassifier,
        .onDeviceObjectCustomProminentNoClassifier,
        .onDeviceObjectCustomProminentWithClassifier,
        .onDeviceObjectCustomMultipleNoClassifier,
        .onDeviceObjectCustomMultipleWithClassifier,
        .pose,
        .poseAccurate,
        .segmentationSelfie,
    ]
    
    private var currentDetector: Detector = .onDeviceFace
    private var isUsingFrontCamera = true
    private var previewLayer: AVCaptureVideoPreviewLayer!
    private lazy var captureSession = AVCaptureSession()
    private lazy var sessionQueue = DispatchQueue(label: Constant.sessionQueueLabel)
    private var lastFrame: CMSampleBuffer?
    private var localImage: UIImage!
    private var userFace: Face? = nil
    private enum Constants {
        static let images = [
            "grace_hopper.jpg", "barcode_128.png", "qr_code.jpg", "beach.jpg",
            "image_has_text.jpg", "liberty.jpg", "bird.jpg",
        ]
        
        static let detectionNoResultsMessage = "No results returned."
        static let failedToDetectObjectsMessage = "Failed to detect objects in image."
        static let localModelFile = (name: "bird", type: "tflite")
        static let labelConfidenceThreshold = 0.75
        static let smallDotRadius: CGFloat = 5.0
        static let largeDotRadius: CGFloat = 10.0
        static let lineColor = UIColor.yellow.cgColor
        static let lineWidth: CGFloat = 3.0
        static let fillColor = UIColor.clear.cgColor
        static let segmentationMaskAlpha: CGFloat = 0.5
    }
    
    private lazy var previewOverlayView: UIImageView = {
        
        precondition(isViewLoaded)
        let previewOverlayView = UIImageView(frame: .zero)
        previewOverlayView.contentMode = UIView.ContentMode.scaleAspectFill
        previewOverlayView.translatesAutoresizingMaskIntoConstraints = false
        return previewOverlayView
    }()
    
    private lazy var annotationOverlayView: UIView = {
        precondition(isViewLoaded)
        let annotationOverlayView = UIView(frame: .zero)
        annotationOverlayView.translatesAutoresizingMaskIntoConstraints = false
        return annotationOverlayView
    }()
    
    /// Initialized when one of the pose detector rows are chosen. Reset to `nil` when neither are.
    private var poseDetector: PoseDetector? = nil
    
    /// Initialized when a segmentation row is chosen. Reset to `nil` otherwise.
    private var segmenter: Segmenter? = nil
    
    var resultsText = ""
    
    /// The detector mode with which detection was most recently run. Only used on the video output
    /// queue. Useful for inferring when to reset detector instances which use a conventional
    /// lifecyle paradigm.
    private var lastDetector: Detector?
    
    // MARK: - IBOutlets
    
    @IBOutlet private weak var cameraView: UIView!
    
    // MARK: - UIViewController
    
    override func viewDidLoad() {
        super.viewDidLoad()
        previewLayer = AVCaptureVideoPreviewLayer(session: captureSession)
        setUpPreviewOverlayView()
        setUpAnnotationOverlayView()
        setUpCaptureSessionOutput()
        setUpCaptureSessionInput()
        processLocalImage()
    }
    
    override func viewDidAppear(_ animated: Bool) {
        super.viewDidAppear(animated)
        
        startSession()
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        
        stopSession()
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        
        previewLayer.frame = cameraView.frame
    }
    
    // MARK: - IBActions
    
    @IBAction func selectDetector(_ sender: Any) {
        presentDetectorsAlertController()
    }
    
    @IBAction func switchCamera(_ sender: Any) {
        isUsingFrontCamera = !isUsingFrontCamera
        removeDetectionAnnotations()
        setUpCaptureSessionInput()
    }
    
    // MARK: On-Device Detections
    
    private func scanBarcodesOnDevice(in image: VisionImage, width: CGFloat, height: CGFloat) {
        // Define the options for a barcode detector.
        let format = BarcodeFormat.all
        let barcodeOptions = BarcodeScannerOptions(formats: format)
        
        // Create a barcode scanner.
        let barcodeScanner = BarcodeScanner.barcodeScanner(options: barcodeOptions)
        var barcodes: [Barcode]
        do {
            barcodes = try barcodeScanner.results(in: image)
        } catch let error {
            print("Failed to scan barcodes with error: \(error.localizedDescription).")
            self.updatePreviewOverlayViewWithLastFrame()
            return
        }
        self.updatePreviewOverlayViewWithLastFrame()
        guard !barcodes.isEmpty else {
            print("Barcode scanner returrned no results.")
            return
        }
        weak var weakSelf = self
        DispatchQueue.main.sync {
            guard let strongSelf = weakSelf else {
                print("Self is nil!")
                return
            }
            for barcode in barcodes {
                let normalizedRect = CGRect(
                    x: barcode.frame.origin.x / width,
                    y: barcode.frame.origin.y / height,
                    width: barcode.frame.size.width / width,
                    height: barcode.frame.size.height / height
                )
                let convertedRect = strongSelf.previewLayer.layerRectConverted(
                    fromMetadataOutputRect: normalizedRect
                )
                UIUtilities.addRectangle(
                    convertedRect,
                    to: strongSelf.annotationOverlayView,
                    color: UIColor.green
                )
                let label = UILabel(frame: convertedRect)
                label.text = barcode.rawValue
                label.adjustsFontSizeToFitWidth = true
                strongSelf.rotate(label, orientation: image.orientation)
                strongSelf.annotationOverlayView.addSubview(label)
            }
        }
    }
    
    
    private func transformMatrix() -> CGAffineTransform {
        guard let image = localImage else { return CGAffineTransform() }
        let imageViewWidth = localImage?.size.width ?? 400
        let imageViewHeight = localImage?.size.height ?? 400
        let imageWidth = image.size.width
        let imageHeight = image.size.height
        
        let imageViewAspectRatio = imageViewWidth / imageViewHeight
        let imageAspectRatio = imageWidth / imageHeight
        let scale =
            (imageViewAspectRatio > imageAspectRatio)
            ? imageViewHeight / imageHeight : imageViewWidth / imageWidth
        
        // Image view's `contentMode` is `scaleAspectFit`, which scales the image to fit the size of the
        // image view by maintaining the aspect ratio. Multiple by `scale` to get image's original size.
        let scaledImageWidth = imageWidth * scale
        let scaledImageHeight = imageHeight * scale
        let xValue = (imageViewWidth - scaledImageWidth) / CGFloat(2.0)
        let yValue = (imageViewHeight - scaledImageHeight) / CGFloat(2.0)
        
        var transform = CGAffineTransform.identity.translatedBy(x: xValue, y: yValue)
        transform = transform.scaledBy(x: scale, y: scale)
        return transform
    }
    
    private func clearResults() {
        removeDetectionAnnotations()
        self.resultsText = ""
    }
    
    private func showResults() {
        let resultsAlertController = UIAlertController(
            title: "Detection Results",
            message: nil,
            preferredStyle: .actionSheet
        )
        resultsAlertController.addAction(
            UIAlertAction(title: "OK", style: .destructive) { _ in
                resultsAlertController.dismiss(animated: true, completion: nil)
            }
        )
        resultsAlertController.message = resultsText
        resultsAlertController.popoverPresentationController?.sourceView = self.view
        present(resultsAlertController, animated: true, completion: nil)
        print(resultsText)
    }
    
    private func processLocalImage() {
        
        let defaults = UserDefaults.standard
        let localStorage = defaults.object(forKey: "LocalImage") as Any
        localImage = UIImage(data: localStorage as! Data)
        print("localVisionImage: \(localImage).")
        
        
        guard (localImage != nil) else {return}
        
        // Initialize a `VisionImage` object with the given `UIImage`.
        let localVisionImage = VisionImage(image: localImage)
        localVisionImage.orientation = localImage!.imageOrientation
        
        print("localVisionImage: \(localVisionImage).")
        
        // When performing latency tests to determine ideal detection settings, run the app in 'release'
        // mode to get accurate performance metrics.
        let options = FaceDetectorOptions()
        options.landmarkMode = .all
        options.classificationMode = .all
        options.performanceMode = .accurate
        options.contourMode = .all
        let faceDetector = FaceDetector.faceDetector(options: options)
        
        weak var weakSelf = self
        faceDetector.process(localVisionImage) { [self] faces, error in
            guard let strongSelf = weakSelf else {
                print("Self is nil!")
                return
            }
            guard error == nil, let faces = faces, !faces.isEmpty else {
                // [START_EXCLUDE]
                let errorString = error?.localizedDescription ?? Constants.detectionNoResultsMessage
                strongSelf.resultsText = "On-Device face detection failed with error: \(errorString)"
                strongSelf.showResults()
                // [END_EXCLUDE]
                return
            }
            
            // Faces detected
            print("faces: \(faces.count).")
            faces.forEach { face in
                print("process face \(face)")
                print("process contours \(face.contours)")
                print("process trackingID \(face.trackingID)")
                print("process hasTrackingID \(face.hasTrackingID)")
                
                if(face.contours.count > 0){
                    userFace = face
                }
                
                let transform = strongSelf.transformMatrix()
                let transformedRect = face.frame.applying(transform)
                UIUtilities.addRectangle(
                    transformedRect,
                    to: strongSelf.annotationOverlayView,
                    color: UIColor.green
                )
            }
            //      strongSelf.resultsText = faces.map { face in
            //        //asdf FACE DETECTED
            //
            //        let headEulerAngleX = face.hasHeadEulerAngleX ? face.headEulerAngleX.description : "NA"
            //        let headEulerAngleY = face.hasHeadEulerAngleY ? face.headEulerAngleY.description : "NA"
            //        let headEulerAngleZ = face.hasHeadEulerAngleZ ? face.headEulerAngleZ.description : "NA"
            //        let leftEyeOpenProbability =
            //          face.hasLeftEyeOpenProbability
            //          ? face.leftEyeOpenProbability.description : "NA"
            //        let rightEyeOpenProbability =
            //          face.hasRightEyeOpenProbability
            //          ? face.rightEyeOpenProbability.description : "NA"
            //        let smilingProbability =
            //          face.hasSmilingProbability
            //          ? face.smilingProbability.description : "NA"
            //        let output = """
            //          Frame: \(face.frame)
            //          Head Euler Angle X: \(headEulerAngleX)
            //          Head Euler Angle Y: \(headEulerAngleY)
            //          Head Euler Angle Z: \(headEulerAngleZ)
            //          Left Eye Open Probability: \(leftEyeOpenProbability)
            //          Right Eye Open Probability: \(rightEyeOpenProbability)
            //          Smiling Probability: \(smilingProbability)
            //          """
            //        return "\(output)"
            //      }.joined(separator: "\n")
            //      strongSelf.showResults()
            // [END_EXCLUDE]
        }
        
        print("updatecd local face: \(userFace != nil).")
    }
    
    private func detectFacesOnDevice(in visionImage: VisionImage, width: CGFloat, height: CGFloat) {
        // When performing latency tests to determine ideal detection settings, run the app in 'release'
        // mode to get accurate performance metrics.
        let options = FaceDetectorOptions()
        options.landmarkMode = .all
        options.classificationMode = .all
        options.performanceMode = .accurate
        options.contourMode = .all
        let faceDetector = FaceDetector.faceDetector(options: options)
        
        weak var weakSelf = self
        var faces: [Face]
        do {
            faces = try faceDetector.results(in: visionImage)
        } catch let error {
            print("Failed to detect faces with error: \(error.localizedDescription).")
            self.updatePreviewOverlayViewWithLastFrame()
            return
        }
        self.updatePreviewOverlayViewWithLastFrame()
        guard !faces.isEmpty else {
            print("On-Device face detector returned no results.")
            return
        }
        
        DispatchQueue.main.sync {
            guard let strongSelf = weakSelf else {
                print("Self is nil!")
                return
            }
            
            var matchingResult : Int = 0
            for face: Face in faces {
                //asdf FACE DETECTED
                print("face.contours")
                print("face.contours.count \(face.contours.count)")
                print(" ")
                print("----------userface----------")
                print(userFace ?? "nil on userface")
                print("-----------------------")
                print(" ")
                print("----------userFace.contours----------")
                print(userFace?.contours ?? "nil on userFace.contours")
                print("-----------------------")
                print(" ")
                print("----------face----------")
                print(face)
                print("-----------------------")
                
                
                if(face.contours.count > 0 && userFace?.contours.count ?? 0 > 0){
                    var distance : Int = 0;
                    for i in 0..<face.contours[0].points.count {
                        let userFaceContour = userFace?.contours[i].points[0]
                        let faceContour = face.contours[i].points[0]
                        
                        
                        print("-----------------------")
                        print("userFaceContour \(userFaceContour)")
                        print("-----------------------")
                        print("-----------------------")
                        print("faceContour \(faceContour)")
                        print("-----------------------")
//                        var diff = Int(faceContour) - Int(userFaceContour)
                        var diff = 1
                        distance += diff * diff;
                    }
                    
                    distance = Int(sqrt(Double(distance)));
                    if (matchingResult == 0 || distance < matchingResult) {
                        matchingResult = distance
                    }
                }
                
                
                let normalizedRect = CGRect(
                    x: face.frame.origin.x / width,
                    y: face.frame.origin.y / height,
                    width: face.frame.size.width / width,
                    height: face.frame.size.height / height
                )
                let standardizedRect = strongSelf.previewLayer.layerRectConverted(
                    fromMetadataOutputRect: normalizedRect
                ).standardized
                UIUtilities.addRectangle(
                    standardizedRect,
                    to: strongSelf.annotationOverlayView,
                    color: UIColor.green
                )
                strongSelf.addContours(for: face, width: width, height: height)
            }
        }
    }
    
    private func detectPose(in image: MLImage, width: CGFloat, height: CGFloat) {
        if let poseDetector = self.poseDetector {
            var poses: [Pose]
            do {
                poses = try poseDetector.results(in: image)
            } catch let error {
                print("Failed to detect poses with error: \(error.localizedDescription).")
                self.updatePreviewOverlayViewWithLastFrame()
                return
            }
            self.updatePreviewOverlayViewWithLastFrame()
            guard !poses.isEmpty else {
                print("Pose detector returned no results.")
                return
            }
            weak var weakSelf = self
            DispatchQueue.main.sync {
                guard let strongSelf = weakSelf else {
                    print("Self is nil!")
                    return
                }
                // Pose detected. Currently, only single person detection is supported.
                poses.forEach { pose in
                    let poseOverlayView = UIUtilities.createPoseOverlayView(
                        forPose: pose,
                        inViewWithBounds: strongSelf.annotationOverlayView.bounds,
                        lineWidth: Constant.lineWidth,
                        dotRadius: Constant.smallDotRadius,
                        positionTransformationClosure: { (position) -> CGPoint in
                            return strongSelf.normalizedPoint(
                                fromVisionPoint: position, width: width, height: height)
                        }
                    )
                    strongSelf.annotationOverlayView.addSubview(poseOverlayView)
                }
            }
        }
    }
    
    private func detectSegmentationMask(in image: VisionImage, sampleBuffer: CMSampleBuffer) {
        guard let segmenter = self.segmenter else {
            return
        }
        var mask: SegmentationMask
        do {
            mask = try segmenter.results(in: image)
        } catch let error {
            print("Failed to perform segmentation with error: \(error.localizedDescription).")
            self.updatePreviewOverlayViewWithLastFrame()
            return
        }
        weak var weakSelf = self
        DispatchQueue.main.sync {
            guard let strongSelf = weakSelf else {
                print("Self is nil!")
                return
            }
            strongSelf.removeDetectionAnnotations()
            
            guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
                print("Failed to get image buffer from sample buffer.")
                return
            }
            
            UIUtilities.applySegmentationMask(
                mask: mask, to: imageBuffer,
                backgroundColor: UIColor.purple.withAlphaComponent(Constant.segmentationMaskAlpha),
                foregroundColor: nil)
            strongSelf.updatePreviewOverlayViewWithImageBuffer(imageBuffer)
        }
        
    }
    
    private func recognizeTextOnDevice(in image: VisionImage, width: CGFloat, height: CGFloat) {
        var recognizedText: Text
        do {
            recognizedText = try TextRecognizer.textRecognizer().results(in: image)
        } catch let error {
            print("Failed to recognize text with error: \(error.localizedDescription).")
            self.updatePreviewOverlayViewWithLastFrame()
            return
        }
        self.updatePreviewOverlayViewWithLastFrame()
        weak var weakSelf = self
        DispatchQueue.main.sync {
            guard let strongSelf = weakSelf else {
                print("Self is nil!")
                return
            }
            
            // Blocks.
            for block in recognizedText.blocks {
                let points = strongSelf.convertedPoints(
                    from: block.cornerPoints, width: width, height: height)
                UIUtilities.addShape(
                    withPoints: points,
                    to: strongSelf.annotationOverlayView,
                    color: UIColor.purple
                )
                
                // Lines.
                for line in block.lines {
                    let points = strongSelf.convertedPoints(
                        from: line.cornerPoints, width: width, height: height)
                    UIUtilities.addShape(
                        withPoints: points,
                        to: strongSelf.annotationOverlayView,
                        color: UIColor.orange
                    )
                    
                    // Elements.
                    for element in line.elements {
                        let normalizedRect = CGRect(
                            x: element.frame.origin.x / width,
                            y: element.frame.origin.y / height,
                            width: element.frame.size.width / width,
                            height: element.frame.size.height / height
                        )
                        let convertedRect = strongSelf.previewLayer.layerRectConverted(
                            fromMetadataOutputRect: normalizedRect
                        )
                        UIUtilities.addRectangle(
                            convertedRect,
                            to: strongSelf.annotationOverlayView,
                            color: UIColor.green
                        )
                        let label = UILabel(frame: convertedRect)
                        label.text = element.text
                        label.adjustsFontSizeToFitWidth = true
                        strongSelf.rotate(label, orientation: image.orientation)
                        strongSelf.annotationOverlayView.addSubview(label)
                    }
                }
            }
        }
    }
    
    private func detectLabels(
        in visionImage: VisionImage,
        width: CGFloat,
        height: CGFloat,
        shouldUseCustomModel: Bool
    ) {
        var options: CommonImageLabelerOptions!
        if shouldUseCustomModel {
            guard
                let localModelFilePath = Bundle.main.path(
                    forResource: Constant.localModelFile.name,
                    ofType: Constant.localModelFile.type
                )
            else {
                print("On-Device label detection failed because custom model was not found.")
                return
            }
            let localModel = LocalModel(path: localModelFilePath)
            options = CustomImageLabelerOptions(localModel: localModel)
        } else {
            options = ImageLabelerOptions()
        }
        options.confidenceThreshold = NSNumber(floatLiteral: Constant.labelConfidenceThreshold)
        let onDeviceLabeler = ImageLabeler.imageLabeler(options: options)
        let labels: [ImageLabel]
        do {
            labels = try onDeviceLabeler.results(in: visionImage)
        } catch let error {
            let errorString = error.localizedDescription
            print("On-Device label detection failed with error: \(errorString)")
            self.updatePreviewOverlayViewWithLastFrame()
            return
        }
        let resultsText = labels.map { label -> String in
            return "Label: \(label.text), Confidence: \(label.confidence), Index: \(label.index)"
        }.joined(separator: "\n")
        
        self.updatePreviewOverlayViewWithLastFrame()
        guard resultsText.count != 0 else { return }
        
        weak var weakSelf = self
        DispatchQueue.main.sync {
            guard let strongSelf = weakSelf else {
                print("Self is nil!")
                return
            }
            let frame = strongSelf.view.frame
            let normalizedRect = CGRect(
                x: Constant.imageLabelResultFrameX,
                y: Constant.imageLabelResultFrameY,
                width: Constant.imageLabelResultFrameWidth,
                height: Constant.imageLabelResultFrameHeight
            )
            let standardizedRect = strongSelf.previewLayer.layerRectConverted(
                fromMetadataOutputRect: normalizedRect
            ).standardized
            UIUtilities.addRectangle(
                standardizedRect,
                to: strongSelf.annotationOverlayView,
                color: UIColor.gray
            )
            let uiLabel = UILabel(frame: standardizedRect)
            uiLabel.text = resultsText
            uiLabel.numberOfLines = 0
            uiLabel.adjustsFontSizeToFitWidth = true
            strongSelf.rotate(uiLabel, orientation: visionImage.orientation)
            strongSelf.annotationOverlayView.addSubview(uiLabel)
        }
    }
    
    private func detectObjectsOnDevice(
        in image: VisionImage,
        width: CGFloat,
        height: CGFloat,
        options: CommonObjectDetectorOptions
    ) {
        let detector = ObjectDetector.objectDetector(options: options)
        var objects: [Object]
        do {
            objects = try detector.results(in: image)
        } catch let error {
            print("Failed to detect objects with error: \(error.localizedDescription).")
            self.updatePreviewOverlayViewWithLastFrame()
            return
        }
        self.updatePreviewOverlayViewWithLastFrame()
        guard !objects.isEmpty else {
            print("On-Device object detector returned no results.")
            return
        }
        
        weak var weakSelf = self
        DispatchQueue.main.sync {
            guard let strongSelf = weakSelf else {
                print("Self is nil!")
                return
            }
            for object in objects {
                let normalizedRect = CGRect(
                    x: object.frame.origin.x / width,
                    y: object.frame.origin.y / height,
                    width: object.frame.size.width / width,
                    height: object.frame.size.height / height
                )
                let standardizedRect = strongSelf.previewLayer.layerRectConverted(
                    fromMetadataOutputRect: normalizedRect
                ).standardized
                UIUtilities.addRectangle(
                    standardizedRect,
                    to: strongSelf.annotationOverlayView,
                    color: UIColor.green
                )
                let label = UILabel(frame: standardizedRect)
                var description = ""
                if let trackingID = object.trackingID {
                    description += "Object ID: " + trackingID.stringValue + "\n"
                }
                description += object.labels.enumerated().map { (index, label) in
                    "Label \(index): \(label.text), \(label.confidence), \(label.index)"
                }.joined(separator: "\n")
                
                label.text = description
                label.numberOfLines = 0
                label.adjustsFontSizeToFitWidth = true
                strongSelf.rotate(label, orientation: image.orientation)
                strongSelf.annotationOverlayView.addSubview(label)
            }
        }
    }
    
    // MARK: - Private
    
    private func setUpCaptureSessionOutput() {
        weak var weakSelf = self
        sessionQueue.async {
            guard let strongSelf = weakSelf else {
                print("Self is nil!")
                return
            }
            strongSelf.captureSession.beginConfiguration()
            // When performing latency tests to determine ideal capture settings,
            // run the app in 'release' mode to get accurate performance metrics
            strongSelf.captureSession.sessionPreset = AVCaptureSession.Preset.medium
            
            let output = AVCaptureVideoDataOutput()
            output.videoSettings = [
                (kCVPixelBufferPixelFormatTypeKey as String): kCVPixelFormatType_32BGRA
            ]
            output.alwaysDiscardsLateVideoFrames = true
            let outputQueue = DispatchQueue(label: Constant.videoDataOutputQueueLabel)
            output.setSampleBufferDelegate(strongSelf, queue: outputQueue)
            guard strongSelf.captureSession.canAddOutput(output) else {
                print("Failed to add capture session output.")
                return
            }
            strongSelf.captureSession.addOutput(output)
            strongSelf.captureSession.commitConfiguration()
        }
    }
    
    private func setUpCaptureSessionInput() {
        weak var weakSelf = self
        sessionQueue.async {
            guard let strongSelf = weakSelf else {
                print("Self is nil!")
                return
            }
            let cameraPosition: AVCaptureDevice.Position = strongSelf.isUsingFrontCamera ? .front : .back
            guard let device = strongSelf.captureDevice(forPosition: cameraPosition) else {
                print("Failed to get capture device for camera position: \(cameraPosition)")
                return
            }
            do {
                strongSelf.captureSession.beginConfiguration()
                let currentInputs = strongSelf.captureSession.inputs
                for input in currentInputs {
                    strongSelf.captureSession.removeInput(input)
                }
                
                let input = try AVCaptureDeviceInput(device: device)
                guard strongSelf.captureSession.canAddInput(input) else {
                    print("Failed to add capture session input.")
                    return
                }
                strongSelf.captureSession.addInput(input)
                strongSelf.captureSession.commitConfiguration()
            } catch {
                print("Failed to create capture device input: \(error.localizedDescription)")
            }
        }
    }
    
    private func startSession() {
        weak var weakSelf = self
        sessionQueue.async {
            guard let strongSelf = weakSelf else {
                print("Self is nil!")
                return
            }
            strongSelf.captureSession.startRunning()
        }
    }
    
    private func stopSession() {
        weak var weakSelf = self
        sessionQueue.async {
            guard let strongSelf = weakSelf else {
                print("Self is nil!")
                return
            }
            strongSelf.captureSession.stopRunning()
        }
    }
    
    private func setUpPreviewOverlayView() {
        cameraView.addSubview(previewOverlayView)
        NSLayoutConstraint.activate([
            previewOverlayView.centerXAnchor.constraint(equalTo: cameraView.centerXAnchor),
            previewOverlayView.centerYAnchor.constraint(equalTo: cameraView.centerYAnchor),
            previewOverlayView.leadingAnchor.constraint(equalTo: cameraView.leadingAnchor),
            previewOverlayView.trailingAnchor.constraint(equalTo: cameraView.trailingAnchor),
            
        ])
    }
    
    private func setUpAnnotationOverlayView() {
        cameraView.addSubview(annotationOverlayView)
        NSLayoutConstraint.activate([
            annotationOverlayView.topAnchor.constraint(equalTo: cameraView.topAnchor),
            annotationOverlayView.leadingAnchor.constraint(equalTo: cameraView.leadingAnchor),
            annotationOverlayView.trailingAnchor.constraint(equalTo: cameraView.trailingAnchor),
            annotationOverlayView.bottomAnchor.constraint(equalTo: cameraView.bottomAnchor),
        ])
    }
    
    private func captureDevice(forPosition position: AVCaptureDevice.Position) -> AVCaptureDevice? {
        if #available(iOS 10.0, *) {
            let discoverySession = AVCaptureDevice.DiscoverySession(
                deviceTypes: [.builtInWideAngleCamera],
                mediaType: .video,
                position: .unspecified
            )
            return discoverySession.devices.first { $0.position == position }
        }
        return nil
    }
    
    private func presentDetectorsAlertController() {
        let alertController = UIAlertController(
            title: Constant.alertControllerTitle,
            message: Constant.alertControllerMessage,
            preferredStyle: .alert
        )
        weak var weakSelf = self
        detectors.forEach { detectorType in
            let action = UIAlertAction(title: detectorType.rawValue, style: .default) {
                [unowned self] (action) in
                guard let value = action.title else { return }
                guard let detector = Detector(rawValue: value) else { return }
                guard let strongSelf = weakSelf else {
                    print("Self is nil!")
                    return
                }
                strongSelf.currentDetector = detector
                strongSelf.removeDetectionAnnotations()
            }
            if detectorType.rawValue == self.currentDetector.rawValue { action.isEnabled = false }
            alertController.addAction(action)
        }
        alertController.addAction(UIAlertAction(title: Constant.cancelActionTitleText, style: .cancel))
        present(alertController, animated: true)
    }
    
    private func removeDetectionAnnotations() {
        for annotationView in annotationOverlayView.subviews {
            annotationView.removeFromSuperview()
        }
    }
    
    private func updatePreviewOverlayViewWithLastFrame() {
        weak var weakSelf = self
        DispatchQueue.main.sync {
            guard let strongSelf = weakSelf else {
                print("Self is nil!")
                return
            }
            
            guard let lastFrame = lastFrame,
                  let imageBuffer = CMSampleBufferGetImageBuffer(lastFrame)
            else {
                return
            }
            strongSelf.updatePreviewOverlayViewWithImageBuffer(imageBuffer)
            strongSelf.removeDetectionAnnotations()
        }
    }
    
    private func updatePreviewOverlayViewWithImageBuffer(_ imageBuffer: CVImageBuffer?) {
        guard let imageBuffer = imageBuffer else {
            return
        }
        let orientation: UIImage.Orientation = isUsingFrontCamera ? .leftMirrored : .right
        let image = UIUtilities.createUIImage(from: imageBuffer, orientation: orientation)
        previewOverlayView.image = image
    }
    
    private func convertedPoints(
        from points: [NSValue]?,
        width: CGFloat,
        height: CGFloat
    ) -> [NSValue]? {
        return points?.map {
            let cgPointValue = $0.cgPointValue
            let normalizedPoint = CGPoint(x: cgPointValue.x / width, y: cgPointValue.y / height)
            let cgPoint = previewLayer.layerPointConverted(fromCaptureDevicePoint: normalizedPoint)
            let value = NSValue(cgPoint: cgPoint)
            return value
        }
    }
    
    private func normalizedPoint(
        fromVisionPoint point: VisionPoint,
        width: CGFloat,
        height: CGFloat
    ) -> CGPoint {
        let cgPoint = CGPoint(x: point.x, y: point.y)
        var normalizedPoint = CGPoint(x: cgPoint.x / width, y: cgPoint.y / height)
        normalizedPoint = previewLayer.layerPointConverted(fromCaptureDevicePoint: normalizedPoint)
        return normalizedPoint
    }
    
    private func addContours(for face: Face, width: CGFloat, height: CGFloat) {
        // Face
        if let faceContour = face.contour(ofType: .face) {
            for point in faceContour.points {
                let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
                UIUtilities.addCircle(
                    atPoint: cgPoint,
                    to: annotationOverlayView,
                    color: UIColor.blue,
                    radius: Constant.smallDotRadius
                )
            }
        }
        
        // Eyebrows
        if let topLeftEyebrowContour = face.contour(ofType: .leftEyebrowTop) {
            for point in topLeftEyebrowContour.points {
                let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
                UIUtilities.addCircle(
                    atPoint: cgPoint,
                    to: annotationOverlayView,
                    color: UIColor.orange,
                    radius: Constant.smallDotRadius
                )
            }
        }
        if let bottomLeftEyebrowContour = face.contour(ofType: .leftEyebrowBottom) {
            for point in bottomLeftEyebrowContour.points {
                let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
                UIUtilities.addCircle(
                    atPoint: cgPoint,
                    to: annotationOverlayView,
                    color: UIColor.orange,
                    radius: Constant.smallDotRadius
                )
            }
        }
        if let topRightEyebrowContour = face.contour(ofType: .rightEyebrowTop) {
            for point in topRightEyebrowContour.points {
                let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
                UIUtilities.addCircle(
                    atPoint: cgPoint,
                    to: annotationOverlayView,
                    color: UIColor.orange,
                    radius: Constant.smallDotRadius
                )
            }
        }
        if let bottomRightEyebrowContour = face.contour(ofType: .rightEyebrowBottom) {
            for point in bottomRightEyebrowContour.points {
                let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
                UIUtilities.addCircle(
                    atPoint: cgPoint,
                    to: annotationOverlayView,
                    color: UIColor.orange,
                    radius: Constant.smallDotRadius
                )
            }
        }
        
        // Eyes
        if let leftEyeContour = face.contour(ofType: .leftEye) {
            for point in leftEyeContour.points {
                let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
                UIUtilities.addCircle(
                    atPoint: cgPoint,
                    to: annotationOverlayView,
                    color: UIColor.cyan,
                    radius: Constant.smallDotRadius
                )
            }
        }
        if let rightEyeContour = face.contour(ofType: .rightEye) {
            for point in rightEyeContour.points {
                let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
                UIUtilities.addCircle(
                    atPoint: cgPoint,
                    to: annotationOverlayView,
                    color: UIColor.cyan,
                    radius: Constant.smallDotRadius
                )
            }
        }
        
        // Lips
        if let topUpperLipContour = face.contour(ofType: .upperLipTop) {
            for point in topUpperLipContour.points {
                let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
                UIUtilities.addCircle(
                    atPoint: cgPoint,
                    to: annotationOverlayView,
                    color: UIColor.red,
                    radius: Constant.smallDotRadius
                )
            }
        }
        if let bottomUpperLipContour = face.contour(ofType: .upperLipBottom) {
            for point in bottomUpperLipContour.points {
                let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
                UIUtilities.addCircle(
                    atPoint: cgPoint,
                    to: annotationOverlayView,
                    color: UIColor.red,
                    radius: Constant.smallDotRadius
                )
            }
        }
        if let topLowerLipContour = face.contour(ofType: .lowerLipTop) {
            for point in topLowerLipContour.points {
                let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
                UIUtilities.addCircle(
                    atPoint: cgPoint,
                    to: annotationOverlayView,
                    color: UIColor.red,
                    radius: Constant.smallDotRadius
                )
            }
        }
        if let bottomLowerLipContour = face.contour(ofType: .lowerLipBottom) {
            for point in bottomLowerLipContour.points {
                let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
                UIUtilities.addCircle(
                    atPoint: cgPoint,
                    to: annotationOverlayView,
                    color: UIColor.red,
                    radius: Constant.smallDotRadius
                )
            }
        }
        
        // Nose
        if let noseBridgeContour = face.contour(ofType: .noseBridge) {
            for point in noseBridgeContour.points {
                let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
                UIUtilities.addCircle(
                    atPoint: cgPoint,
                    to: annotationOverlayView,
                    color: UIColor.yellow,
                    radius: Constant.smallDotRadius
                )
            }
        }
        if let noseBottomContour = face.contour(ofType: .noseBottom) {
            for point in noseBottomContour.points {
                let cgPoint = normalizedPoint(fromVisionPoint: point, width: width, height: height)
                UIUtilities.addCircle(
                    atPoint: cgPoint,
                    to: annotationOverlayView,
                    color: UIColor.yellow,
                    radius: Constant.smallDotRadius
                )
            }
        }
    }
    
    /// Resets any detector instances which use a conventional lifecycle paradigm. This method is
    /// expected to be invoked on the AVCaptureOutput queue - the same queue on which detection is
    /// run.
    private func resetManagedLifecycleDetectors(activeDetector: Detector) {
        if activeDetector == self.lastDetector {
            // Same row as before, no need to reset any detectors.
            return
        }
        // Clear the old detector, if applicable.
        switch self.lastDetector {
        case .pose, .poseAccurate:
            self.poseDetector = nil
            break
        case .segmentationSelfie:
            self.segmenter = nil
            break
        default:
            break
        }
        // Initialize the new detector, if applicable.
        switch activeDetector {
        case .pose, .poseAccurate:
            // The `options.detectorMode` defaults to `.stream`
            let options = activeDetector == .pose ? PoseDetectorOptions() : AccuratePoseDetectorOptions()
            self.poseDetector = PoseDetector.poseDetector(options: options)
            break
        case .segmentationSelfie:
            // The `options.segmenterMode` defaults to `.stream`
            let options = SelfieSegmenterOptions()
            self.segmenter = Segmenter.segmenter(options: options)
            break
        default:
            break
        }
        self.lastDetector = activeDetector
    }
    
    private func rotate(_ view: UIView, orientation: UIImage.Orientation) {
        var degree: CGFloat = 0.0
        switch orientation {
        case .up, .upMirrored:
            degree = 90.0
        case .rightMirrored, .left:
            degree = 180.0
        case .down, .downMirrored:
            degree = 270.0
        case .leftMirrored, .right:
            degree = 0.0
        }
        view.transform = CGAffineTransform.init(rotationAngle: degree * 3.141592654 / 180)
    }
}

// MARK: AVCaptureVideoDataOutputSampleBufferDelegate

extension CameraViewController: AVCaptureVideoDataOutputSampleBufferDelegate {
    
    func captureOutput(
        _ output: AVCaptureOutput,
        didOutput sampleBuffer: CMSampleBuffer,
        from connection: AVCaptureConnection
    ) {
        guard let imageBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else {
            print("Failed to get image buffer from sample buffer.")
            return
        }
        // Evaluate `self.currentDetector` once to ensure consistency throughout this method since it
        // can be concurrently modified from the main thread.
        let activeDetector = self.currentDetector
        resetManagedLifecycleDetectors(activeDetector: activeDetector)
        
        lastFrame = sampleBuffer
        let visionImage = VisionImage(buffer: sampleBuffer)
        let orientation = UIUtilities.imageOrientation(
            fromDevicePosition: isUsingFrontCamera ? .front : .back
        )
        visionImage.orientation = orientation
        
        guard let inputImage = MLImage(sampleBuffer: sampleBuffer) else {
            print("Failed to create MLImage from sample buffer.")
            return
        }
        inputImage.orientation = orientation
        
        let imageWidth = CGFloat(CVPixelBufferGetWidth(imageBuffer))
        let imageHeight = CGFloat(CVPixelBufferGetHeight(imageBuffer))
        var shouldEnableClassification = false
        var shouldEnableMultipleObjects = false
        switch activeDetector {
        case .onDeviceObjectProminentWithClassifier, .onDeviceObjectMultipleWithClassifier,
             .onDeviceObjectCustomProminentWithClassifier, .onDeviceObjectCustomMultipleWithClassifier:
            shouldEnableClassification = true
        default:
            break
        }
        switch activeDetector {
        case .onDeviceObjectMultipleNoClassifier, .onDeviceObjectMultipleWithClassifier,
             .onDeviceObjectCustomMultipleNoClassifier, .onDeviceObjectCustomMultipleWithClassifier:
            shouldEnableMultipleObjects = true
        default:
            break
        }
        
        switch activeDetector {
        case .onDeviceBarcode:
            scanBarcodesOnDevice(in: visionImage, width: imageWidth, height: imageHeight)
        case .onDeviceFace:
            detectFacesOnDevice(in: visionImage, width: imageWidth, height: imageHeight)
        case .onDeviceText:
            recognizeTextOnDevice(in: visionImage, width: imageWidth, height: imageHeight)
        case .onDeviceImageLabel:
            detectLabels(
                in: visionImage, width: imageWidth, height: imageHeight, shouldUseCustomModel: false)
        case .onDeviceImageLabelsCustom:
            detectLabels(
                in: visionImage, width: imageWidth, height: imageHeight, shouldUseCustomModel: true)
        case .onDeviceObjectProminentNoClassifier, .onDeviceObjectProminentWithClassifier,
             .onDeviceObjectMultipleNoClassifier, .onDeviceObjectMultipleWithClassifier:
            // The `options.detectorMode` defaults to `.stream`
            let options = ObjectDetectorOptions()
            options.shouldEnableClassification = shouldEnableClassification
            options.shouldEnableMultipleObjects = shouldEnableMultipleObjects
            detectObjectsOnDevice(
                in: visionImage,
                width: imageWidth,
                height: imageHeight,
                options: options)
        case .onDeviceObjectCustomProminentNoClassifier, .onDeviceObjectCustomProminentWithClassifier,
             .onDeviceObjectCustomMultipleNoClassifier, .onDeviceObjectCustomMultipleWithClassifier:
            guard
                let localModelFilePath = Bundle.main.path(
                    forResource: Constant.localModelFile.name,
                    ofType: Constant.localModelFile.type
                )
            else {
                print("Failed to find custom local model file.")
                return
            }
            let localModel = LocalModel(path: localModelFilePath)
            // The `options.detectorMode` defaults to `.stream`
            let options = CustomObjectDetectorOptions(localModel: localModel)
            options.shouldEnableClassification = shouldEnableClassification
            options.shouldEnableMultipleObjects = shouldEnableMultipleObjects
            detectObjectsOnDevice(
                in: visionImage,
                width: imageWidth,
                height: imageHeight,
                options: options)
            
        case .pose, .poseAccurate:
            detectPose(in: inputImage, width: imageWidth, height: imageHeight)
        case .segmentationSelfie:
            detectSegmentationMask(in: visionImage, sampleBuffer: sampleBuffer)
        }
    }
}

// MARK: - Constants

public enum Detector: String {
    case onDeviceBarcode = "Barcode Scanning"
    case onDeviceFace = "Face Detection"
    case onDeviceText = "Text Recognition"
    case onDeviceImageLabel = "Image Labeling"
    case onDeviceImageLabelsCustom = "Image Labeling Custom"
    case onDeviceObjectProminentNoClassifier = "ODT, single, no labeling"
    case onDeviceObjectProminentWithClassifier = "ODT, single, labeling"
    case onDeviceObjectMultipleNoClassifier = "ODT, multiple, no labeling"
    case onDeviceObjectMultipleWithClassifier = "ODT, multiple, labeling"
    case onDeviceObjectCustomProminentNoClassifier = "ODT, custom, single, no labeling"
    case onDeviceObjectCustomProminentWithClassifier = "ODT, custom, single, labeling"
    case onDeviceObjectCustomMultipleNoClassifier = "ODT, custom, multiple, no labeling"
    case onDeviceObjectCustomMultipleWithClassifier = "ODT, custom, multiple, labeling"
    case pose = "Pose Detection"
    case poseAccurate = "Pose Detection, accurate"
    case segmentationSelfie = "Selfie Segmentation"
}

private enum Constant {
    static let alertControllerTitle = "Vision Detectors"
    static let alertControllerMessage = "Select a detector"
    static let cancelActionTitleText = "Cancel"
    static let videoDataOutputQueueLabel = "com.google.mlkit.visiondetector.VideoDataOutputQueue"
    static let sessionQueueLabel = "com.google.mlkit.visiondetector.SessionQueue"
    static let noResultsMessage = "No Results"
    static let localModelFile = (name: "bird", type: "tflite")
    static let labelConfidenceThreshold = 0.75
    static let smallDotRadius: CGFloat = 4.0
    static let lineWidth: CGFloat = 3.0
    static let originalScale: CGFloat = 1.0
    static let padding: CGFloat = 10.0
    static let resultsLabelHeight: CGFloat = 200.0
    static let resultsLabelLines = 5
    static let imageLabelResultFrameX = 0.4
    static let imageLabelResultFrameY = 0.1
    static let imageLabelResultFrameWidth = 0.5
    static let imageLabelResultFrameHeight = 0.8
    static let segmentationMaskAlpha: CGFloat = 0.5
}



