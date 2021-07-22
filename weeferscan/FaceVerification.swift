

import Foundation
import TensorFlowLite
import AVFoundation
import CoreVideo
import MLImage
import MLKit

class ValidationFace : InterfaceValidation {

    private var OUTPUT_SIZE : Int = 192;
    private var IMAGE_MEAN : Float = 128.0;
    private var IMAGE_STD : Float = 128.0;
    private var registered = [String : ModelFace]();
    private var outputFloat : [[Float]] = [[1],[192]];
//    private var labels: Vector<String>  = new Vector<>();

    private imgData: CMSampleBuffer
    private var interpreter : Interpreter? = nil;
    private var isModelQuantized: Bool = false;
    private var inputSize: Int = 0;
    private var intValues: [Int] = [];
    private var lastFrame: CMSampleBuffer?;
    private var localImage: UIImage!
    private var userFace: Face? = nil;

    func register(name: String, modelFace: ModelFace) {
        registered[name] = modelFace
    }

    func loadModelFile (modelFilename: String ) -> CMSampleBuffer{
        AssetFileDescriptor fileDescriptor = assets.openFd(modelFilename);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
    
    
    func create(
        modelFilename: String ,
        labelFilename: String ,
        inputSize: Int ,
        isQuantized:Bool
    ) -> InterfaceValidation {
        let validationFace: ValidationFace = ValidationFace();

//        String actualFilename = labelFilename.split("file:///android_asset/")[1];
//        InputStream labelsInput = assetManager.open(actualFilename);
//        BufferedReader br = new BufferedReader(new InputStreamReader(labelsInput));
//        String line;
//        while ((line = br.readLine()) != null) {
//            validationFace.labels.add(line);
//        }
//        br.close();
    let defaults = UserDefaults.standard
    let localStorage = defaults.object(forKey: "LocalImage") as Any
    localImage = UIImage(data: localStorage as! Data)
    print("localVisionImage: \(localImage).")
    
    
    // Initialize a `VisionImage` object with the given `UIImage`.
    let localVisionImage = VisionImage(image: localImage)
    localVisionImage.orientation = localImage!.imageOrientation


        validationFace.inputSize = inputSize;
        do {
//            validationFace.interpreter = new Interpreter(loadModelFile(assetManager, modelFilename));
            interpreter = try Interpreter.init(modelPath: modelFilename)
        } catch {
            print("Interpreter.init: \(error).")
        }
        

        validationFace.isModelQuantized = isQuantized;
        // Pre-allocate buffers.
        var numBytesPerChannel: Int;
        if (isQuantized) {
            numBytesPerChannel = 1; // Quantized
        } else {
            numBytesPerChannel = 4; // Floating point
        }
        validationFace.imgData = Byte ByteBuffer.allocateDirect(1 * validationFace.inputSize * validationFace.inputSize * 3 * numBytesPerChannel);
        validationFace.imgData.order(ByteOrder.nativeOrder());
        validationFace.intValues = new int[validationFace.inputSize * validationFace.inputSize];
        return validationFace;
    }

    func findNearest(emb: [Float] ) -> [String : Float] {

        var ret: [String : Float] = [String : Float]();
        
        for (e, entry: ModelFace) in registered {
            var name : String = entry.getKey();
            var knownEmb: [Float] = (([[Float]]) entry.getValue().getExtra())[0];
            
            for (key, value) in companies {
                print("\(key) -> \(value)")
            }

            var distance: Float = 0;
            for (index, element) in emb {
                var dif: Float = emb[index] - knownEmb[index];
                distance += diff * diff;
            }
            
            
            distance = Float(distance.squareRoot());
            if (ret == nil || distance < ret.second) {
                ret = [name: distance];
            }
        }
        return ret;
    }
    
    func recognizeImage(bitmap: Any, getExtra: Bool) -> [ModelFace] {
        outputFloat = [[1][OUTPUT_SIZE]];

        if (!storeExtra) {
            tensorCamera(bitmap);
        } else {
            tensorImageStorage();
        }

        float distance = Float.MAX_VALUE;
        String id = "0";
        String label = "?";

        if (registered.size() > 0) {
            final Pair<String, Float> nearest = findNearest(outputFloat[0]);
            if (nearest != null) {
                label = nearest.first;
                distance = nearest.second;
            }
        }

        final int numDetectionsOutput = 1;
        final ArrayList<ModelFace> modelFaces = new ArrayList<>(numDetectionsOutput);
        ModelFace rec = new ModelFace(
                id,
                label,
                distance,
                new RectF());

        modelFaces.add(rec);
        if (storeExtra) {
            rec.setExtra(outputFloat);
        }
        Trace.endSection();
        return modelFaces;
    }

    func tensorCamera(bitmap: UIImage) {
        // Log this method so that it can be analyzed with systrace.
        Trace.beginSection("recognizeImage");

        Trace.beginSection("preprocessBitmap");
        // Preprocess the image data from 0-255 int to normalized float based
        // on the provided parameters.
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());

        imgData.rewind();
        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                int pixelValue = intValues[i * inputSize + j];
                if (isModelQuantized) {
                    // Quantized model
                    imgData.put((byte) ((pixelValue >> 16) & 0xFF));
                    imgData.put((byte) ((pixelValue >> 8) & 0xFF));
                    imgData.put((byte) (pixelValue & 0xFF));
                } else { // Float model
                    imgData.putFloat((((pixelValue >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat((((pixelValue >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                    imgData.putFloat(((pixelValue & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                }
            }
        }
        Trace.endSection();

        Trace.beginSection("feed");
        Object[] inputArray = {imgData};
        Trace.endSection();

        var outputMap:  [Int, NSObject] = [Int, NSObject]()
        outputMap.put(0, outputFloat);

        Trace.beginSection("run");
        interpreter.runForMultipleInputsOutputs(inputArray, outputMap);
        Trace.endSection();
    }

    private void tensorImageStorage() {
        File imgFile = new File("/sdcard/Download/user.png");
        if (imgFile.exists()) {
            Bitmap bitmapStorage = BitmapFactory.decodeFile(imgFile.getAbsolutePath());

            FaceDetectorOptions faceDetectorOptions =
                    new FaceDetectorOptions.Builder()
                            .setPerformanceMode(FaceDetectorOptions.PERFORMANCE_MODE_FAST)
                            .setContourMode(FaceDetectorOptions.LANDMARK_MODE_NONE)
                            .setClassificationMode(FaceDetectorOptions.CLASSIFICATION_MODE_NONE)
                            .build();
            FaceDetector imageDetector = FaceDetection.getClient(faceDetectorOptions);

            InputImage image = InputImage.fromBitmap(bitmapStorage, 0);
            imageDetector
                    .process(image)
                    .addOnSuccessListener(faces -> {
                        if (faces.size() > 0) {
                            Gson gson = new Gson();

                            Face face = faces.get(0);


                            final RectF boundingBox = new RectF(face.getBoundingBox());
                            if (boundingBox != null) {
                                RectF faceBB = new RectF(boundingBox);
                                Bitmap crop = Bitmap.createBitmap(bitmapStorage,
                                        (int) faceBB.left,
                                        (int) faceBB.top,
                                        (int) faceBB.width(),
                                        (int) faceBB.height());
                                HelperFace helperFace = new HelperFace();
                                ByteBuffer byteBuffer = helperFace.convertBitmapToBuffer(crop);

                                Trace.beginSection("feed");
                                Object[] inputArray = {byteBuffer};
                                Trace.endSection();

                                Map<Integer, Object> outputMap = new HashMap<>();
                                outputMap.put(0, outputFloat);

                                Trace.beginSection("run");
                                interpreter.runForMultipleInputsOutputs(inputArray, outputMap);
                                Trace.endSection();
                                imageDetector.close();
                            }
                        }
                    });
        }
    }
    
    func enableStatLogging(debug: Bool) {
        <#code#>
    }
    
    func close() {
        <#code#>
    }
}
