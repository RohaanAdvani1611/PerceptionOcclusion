import AVFoundation

enum CameraError: LocalizedError {
    case accessDenied, noCamera, cannotConfigure

    var errorDescription: String? {
        switch self {
        case .accessDenied: "Camera access was denied. Allow it in System Settings → Privacy & Security → Camera."
        case .noCamera: "No camera found."
        case .cannotConfigure: "The camera could not be configured."
        }
    }
}

/// Delivers camera frames as 640x480 BGRA pixel buffers on a background queue.
///
/// `alwaysDiscardsLateVideoFrames` drops frames that arrive while the previous one is still
/// being processed, so the app always works on the newest frame instead of falling behind.
nonisolated final class CameraManager: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    /// Frames are delivered, and should be processed, on this serial queue.
    let queue = DispatchQueue(label: "DepthCam.camera", qos: .userInteractive)
    /// Called on `queue` for every frame that isn't dropped. Set before `start()`.
    var onFrame: ((CVPixelBuffer) -> Void)?

    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()

    func start() async throws {
        guard await AVCaptureDevice.requestAccess(for: .video) else { throw CameraError.accessDenied }
        try configure()
        queue.async { self.session.startRunning() }
    }

    private func configure() throws {
        guard let camera = Self.builtInCamera() else { throw CameraError.noCamera }
        let input = try AVCaptureDeviceInput(device: camera)

        session.beginConfiguration()
        guard session.canAddInput(input) else { throw CameraError.cannotConfigure }
        session.addInput(input)

        var settings: [String: Any] = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        #if os(macOS)
        // Only the Mac's video output can scale frames itself; this makes 640x480 certain there
        settings[kCVPixelBufferWidthKey as String] = 640
        settings[kCVPixelBufferHeightKey as String] = 480
        #endif
        output.videoSettings = settings
        output.alwaysDiscardsLateVideoFrames = true
        output.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(output) else { throw CameraError.cannotConfigure }
        session.addOutput(output)
        session.commitConfiguration()

        // 640x480 matches the C++ pipeline's camera frames. The Mac's FaceTime camera ignores
        // the .vga640x480 preset and sends 1920x1080, so the format is chosen directly, after
        // the session is configured so its default preset can't override it.
        if let format = camera.formats.first(where: {
            let size = CMVideoFormatDescriptionGetDimensions($0.formatDescription)
            return size.width == 640 && size.height == 480
        }) {
            try camera.lockForConfiguration()
            camera.activeFormat = format
            camera.unlockForConfiguration()
            #if os(iOS)
            session.sessionPreset = .inputPriority
            #endif
        }
    }

    /// The back camera on iPhone, the FaceTime camera on Mac. Avoids picking an iPhone
    /// connected to the Mac as a Continuity Camera.
    private static func builtInCamera() -> AVCaptureDevice? {
        #if os(iOS)
        AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: .back)
        #else
        AVCaptureDevice.DiscoverySession(deviceTypes: [.builtInWideAngleCamera], mediaType: .video,
                                         position: .unspecified).devices.first
            ?? AVCaptureDevice.default(for: .video)
        #endif
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        onFrame?(pixelBuffer)
    }
}
