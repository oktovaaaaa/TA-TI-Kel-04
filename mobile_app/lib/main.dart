import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint("Inisialisasi kamera awal: $e");
  }
  runApp(const FaceAntiSpoofingApp());
}

class FaceAntiSpoofingApp extends StatelessWidget {
  const FaceAntiSpoofingApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Face Anti-Spoofing',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF121212),
        colorScheme: const ColorScheme.dark(
          primary: Colors.blueAccent,
          secondary: Colors.greenAccent,
          error: Colors.redAccent,
          surface: Color(0xFF1E1E1E),
        ),
      ),
      home: const SimpleScannerScreen(),
    );
  }
}

class SimpleScannerScreen extends StatefulWidget {
  const SimpleScannerScreen({super.key});

  @override
  State<SimpleScannerScreen> createState() => _SimpleScannerScreenState();
}

class _SimpleScannerScreenState extends State<SimpleScannerScreen> {
  CameraController? _cameraController;
  int _selectedCameraIndex = 0;
  bool _isCameraReady = false;
  bool _isProcessing = false;
  String _serverUrl = "http://100.68.116.102:8000";
  bool _isServerConnected = false;
  Map<String, dynamic>? _scanResult;
  File? _lastScannedImage;

  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _initCamera();
    _checkServerHealth();
  }

  Future<void> _initCamera() async {
    try {
      if (cameras.isEmpty) {
        cameras = await availableCameras();
      }
      if (cameras.isNotEmpty) {
        // Cari kamera depan terlebih dahulu
        int frontCam = cameras.indexWhere((c) => c.lensDirection == CameraLensDirection.front);
        _selectedCameraIndex = frontCam != -1 ? frontCam : 0;
        await _startCamera(_selectedCameraIndex);
      } else {
        setState(() => _isCameraReady = false);
      }
    } catch (e) {
      debugPrint("Gagal load camera: $e");
      setState(() => _isCameraReady = false);
    }
  }

  Future<void> _startCamera(int index) async {
    if (cameras.isEmpty || index >= cameras.length) return;

    if (_cameraController != null) {
      await _cameraController!.dispose();
    }

    final controller = CameraController(
      cameras[index],
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.jpeg,
    );

    try {
      await controller.initialize();
      if (!mounted) return;
      setState(() {
        _cameraController = controller;
        _isCameraReady = true;
      });
    } catch (e) {
      debugPrint("Error saat initialize camera: $e");
      if (mounted) setState(() => _isCameraReady = false);
    }
  }

  Future<void> _switchCamera() async {
    if (cameras.length < 2) {
      _showSnackbar("Hanya tersedia 1 kamera pada perangkat ini.");
      return;
    }
    _selectedCameraIndex = (_selectedCameraIndex + 1) % cameras.length;
    await _startCamera(_selectedCameraIndex);
  }

  Future<void> _checkServerHealth() async {
    try {
      final res = await http.get(Uri.parse("$_serverUrl/ping")).timeout(const Duration(seconds: 4));
      if (res.statusCode == 200) {
        if (mounted) setState(() => _isServerConnected = true);
      } else {
        if (mounted) setState(() => _isServerConnected = false);
      }
    } catch (_) {
      if (mounted) setState(() => _isServerConnected = false);
    }
  }

  Future<void> _scanFromCamera() async {
    if (_isProcessing) return;

    if (_cameraController == null || !_cameraController!.value.isInitialized) {
      _showSnackbar("Kamera belum aktif. Silakan pilih foto dari Galeri.");
      return;
    }

    setState(() => _isProcessing = true);

    try {
      // Ambil foto dengan timeout 5 detik agar tidak macet di emulator
      final XFile picture = await _cameraController!.takePicture().timeout(
        const Duration(seconds: 5),
        onTimeout: () => throw TimeoutException("Kamera emulator tidak merespon."),
      );

      final file = File(picture.path);
      setState(() => _lastScannedImage = file);
      await _sendImageToServer(file);
    } catch (e) {
      _showSnackbar("Gagal ambil foto kamera: $e. Coba gunakan tombol Galeri.");
    } finally {
      if (mounted) setState(() => _isProcessing = false);
    }
  }

  Future<void> _pickFromGallery() async {
    if (_isProcessing) return;

    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image == null) return;

    final file = File(image.path);
    setState(() {
      _isProcessing = true;
      _lastScannedImage = file;
    });

    await _sendImageToServer(file);
    if (mounted) setState(() => _isProcessing = false);
  }

  Future<void> _sendImageToServer(File file) async {
    try {
      var request = http.MultipartRequest("POST", Uri.parse("$_serverUrl/predict"));
      request.files.add(await http.MultipartFile.fromPath("image", file.path));

      var streamedResponse = await request.send().timeout(const Duration(seconds: 10));
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (mounted) {
          setState(() {
            _scanResult = data;
            _isServerConnected = true;
          });
        }
      } else {
        _showSnackbar("Server error: kode ${response.statusCode}");
      }
    } catch (e) {
      _showSnackbar("Koneksi gagal ke $_serverUrl. Periksa server/Tailscale.");
      if (mounted) setState(() => _isServerConnected = false);
    }
  }

  void _showSettingsDialog() {
    final controller = TextEditingController(text: _serverUrl);
    showDialog(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text("Pengaturan Server GPU"),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text("Masukkan URL Tailscale Server:", style: TextStyle(fontSize: 13)),
            const SizedBox(height: 10),
            TextField(
              controller: controller,
              decoration: const InputDecoration(
                border: OutlineInputBorder(),
                hintText: "http://100.68.116.102:8000",
              ),
            ),
          ],
        ),
        actions: [
          TextButton(onPressed: () => Navigator.pop(ctx), child: const Text("Batal")),
          ElevatedButton(
            onPressed: () {
              setState(() => _serverUrl = controller.text.trim());
              Navigator.pop(ctx);
              _checkServerHealth();
            },
            child: const Text("Simpan & Tes"),
          ),
        ],
      ),
    );
  }

  void _showSnackbar(String msg) {
    if (!mounted) return;
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(msg), duration: const Duration(seconds: 3)),
    );
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Face Anti-Spoofing", style: TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
        backgroundColor: const Color(0xFF1E1E1E),
        elevation: 0,
        actions: [
          // Badge Status Koneksi Server
          Center(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
              margin: const EdgeInsets.only(right: 8),
              decoration: BoxDecoration(
                color: _isServerConnected ? Colors.green.withValues(alpha: 0.2) : Colors.red.withValues(alpha: 0.2),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: _isServerConnected ? Colors.green : Colors.red),
              ),
              child: Row(
                children: [
                  CircleAvatar(radius: 4, backgroundColor: _isServerConnected ? Colors.green : Colors.red),
                  const SizedBox(width: 6),
                  Text(
                    _isServerConnected ? "ONLINE" : "OFFLINE",
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.bold,
                      color: _isServerConnected ? Colors.greenAccent : Colors.redAccent,
                    ),
                  ),
                ],
              ),
            ),
          ),
          IconButton(
            icon: const Icon(Icons.settings),
            tooltip: "Pengaturan IP",
            onPressed: _showSettingsDialog,
          ),
        ],
      ),
      body: SafeArea(
        child: Column(
          children: [
            // 1. Tampilan Kamera (Polos & Jelas)
            Expanded(
              flex: 5,
              child: Container(
                margin: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.black,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.white24, width: 1.5),
                ),
                clipBehavior: Clip.antiAlias,
                child: Stack(
                  fit: StackFit.expand,
                  children: [
                    if (_isCameraReady && _cameraController != null)
                      CameraPreview(_cameraController!)
                    else
                      Center(
                        child: Padding(
                          padding: const EdgeInsets.all(20),
                          child: Column(
                            mainAxisSize: MainAxisSize.min,
                            children: [
                              const Icon(Icons.videocam_off_outlined, size: 48, color: Colors.white54),
                              const SizedBox(height: 12),
                              const Text(
                                "Kamera Emulator Belum Aktif",
                                style: TextStyle(fontWeight: FontWeight.bold, fontSize: 15),
                              ),
                              const SizedBox(height: 6),
                              const Text(
                                "Hubungkan webcam di emulator via (... -> Camera -> Webcam0),\natau gunakan tombol 'Galeri' di bawah untuk uji coba foto.",
                                textAlign: TextAlign.center,
                                style: TextStyle(fontSize: 12, color: Colors.white60),
                              ),
                              const SizedBox(height: 12),
                              OutlinedButton.icon(
                                onPressed: _initCamera,
                                icon: const Icon(Icons.refresh, size: 16),
                                label: const Text("Coba Muat Ulang Kamera"),
                              ),
                            ],
                          ),
                        ),
                      ),

                    // Kotak Panduan Posisi Wajah Simpel (Garis Putih Tipis)
                    Center(
                      child: Container(
                        width: 220,
                        height: 280,
                        decoration: BoxDecoration(
                          borderRadius: BorderRadius.circular(120),
                          border: Border.all(
                            color: _isProcessing ? Colors.blueAccent : Colors.white54,
                            width: 2,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),

            // 2. Banner Hasil Deteksi (Polos & Jelas: LIVE atau SPOOF)
            Expanded(
              flex: 3,
              child: Container(
                width: double.infinity,
                margin: const EdgeInsets.symmetric(horizontal: 12),
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: const Color(0xFF1E1E1E),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: _scanResult == null
                        ? Colors.white12
                        : (_scanResult!["prediction"] == "NO_FACE"
                            ? Colors.amber
                            : (_scanResult!["is_live"] == true ? Colors.green : Colors.red)),
                    width: 2,
                  ),
                ),
                child: _isProcessing
                    ? const Center(
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            SizedBox(
                              width: 24,
                              height: 24,
                              child: CircularProgressIndicator(color: Colors.blueAccent, strokeWidth: 2.5),
                            ),
                            SizedBox(height: 12),
                            Text(
                              "Sedang menguji keaslian wajah di GPU...",
                              style: TextStyle(color: Colors.white70, fontSize: 13),
                            ),
                          ],
                        ),
                      )
                    : (_scanResult == null
                        ? const Center(
                            child: Column(
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Icon(Icons.face_retouching_natural, size: 36, color: Colors.white38),
                                SizedBox(height: 8),
                                Text(
                                  "Belum Ada Pemindaian",
                                  style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16, color: Colors.white70),
                                ),
                                SizedBox(height: 4),
                                Text(
                                  "Tekan 'Ambil Foto' atau 'Pilih dari Galeri'",
                                  style: TextStyle(fontSize: 12, color: Colors.white38),
                                ),
                              ],
                            ),
                          )
                        : _buildResultView(_scanResult!)),
              ),
            ),

            const SizedBox(height: 10),

            // 3. Tombol Aksi Bawah (Sederhana & Bersih)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
              child: Row(
                children: [
                  // Tombol Galeri
                  Expanded(
                    flex: 2,
                    child: OutlinedButton.icon(
                      style: OutlinedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                      ),
                      onPressed: _isProcessing ? null : _pickFromGallery,
                      icon: const Icon(Icons.photo_library),
                      label: const Text("Galeri"),
                    ),
                  ),
                  const SizedBox(width: 10),

                  // Tombol Utama Ambil Foto / Scan
                  Expanded(
                    flex: 3,
                    child: ElevatedButton.icon(
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.blueAccent,
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                      ),
                      onPressed: _isProcessing ? null : _scanFromCamera,
                      icon: _isProcessing
                          ? const SizedBox(
                              width: 18,
                              height: 18,
                              child: CircularProgressIndicator(color: Colors.white, strokeWidth: 2),
                            )
                          : const Icon(Icons.camera_alt, color: Colors.white),
                      label: Text(
                        _isProcessing ? "MEMPROSES..." : "AMBIL FOTO",
                        style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.white),
                      ),
                    ),
                  ),
                  const SizedBox(width: 10),

                  // Tombol Putar Kamera
                  IconButton.outlined(
                    style: IconButton.styleFrom(
                      padding: const EdgeInsets.all(12),
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(10)),
                    ),
                    onPressed: _isProcessing ? null : _switchCamera,
                    icon: const Icon(Icons.flip_camera_android),
                    tooltip: "Ganti Kamera",
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildResultView(Map<String, dynamic> data) {
    final String prediction = data["prediction"] ?? "";
    final bool faceDetected = data["face_detected"] ?? (prediction != "NO_FACE");

    // Jika TIDAK ADA wajah terdeteksi
    if (!faceDetected || prediction == "NO_FACE") {
      final double latency = (data["latency_ms"] as num?)?.toDouble() ?? 0.0;
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: double.infinity,
            padding: const EdgeInsets.symmetric(vertical: 10),
            decoration: BoxDecoration(
              color: Colors.amber.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.amber, width: 1.5),
            ),
            child: const Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(Icons.person_off_rounded, color: Colors.amber, size: 28),
                SizedBox(width: 8),
                Text(
                  "TIDAK ADA WAJAH TERDETEKSI",
                  style: TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.w900,
                    color: Colors.amber,
                    letterSpacing: 0.5,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 10),
          const Text(
            "Posisikan wajah Anda tepat di dalam bingkai.",
            style: TextStyle(fontSize: 13, color: Colors.white70),
          ),
          const SizedBox(height: 4),
          Text(
            "Waktu deteksi: ${latency.toStringAsFixed(0)} ms",
            style: const TextStyle(fontSize: 11, color: Colors.white38),
          ),
        ],
      );
    }

    final bool isLive = data["is_live"] ?? false;
    final double confLive = (data["confidence_live"] as num?)?.toDouble() ?? 0.0;
    final double confSpoof = (data["confidence_spoof"] as num?)?.toDouble() ?? 0.0;
    final double latency = (data["latency_ms"] as num?)?.toDouble() ?? 0.0;

    final Color badgeColor = isLive ? Colors.green : Colors.red;

    return Column(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        // Label Status Besar
        Container(
          width: double.infinity,
          padding: const EdgeInsets.symmetric(vertical: 10),
          decoration: BoxDecoration(
            color: badgeColor.withValues(alpha: 0.15),
            borderRadius: BorderRadius.circular(8),
            border: Border.all(color: badgeColor, width: 1.5),
          ),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(isLive ? Icons.check_circle : Icons.warning_rounded, color: badgeColor, size: 28),
              const SizedBox(width: 8),
              Text(
                isLive ? "WAJAH ASLI (LIVE)" : "TERDETEKSI SPOOFING!",
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w900,
                  color: badgeColor,
                  letterSpacing: 0.5,
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 12),

        // Rincian Probabilitas & Latensi
        Row(
          children: [
            if (_lastScannedImage != null)
              Container(
                margin: const EdgeInsets.only(right: 12),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(6),
                  child: Image.file(_lastScannedImage!, width: 42, height: 42, fit: BoxFit.cover),
                ),
              ),
            Expanded(
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceAround,
                children: [
                  _buildStatItem("Probabilitas Live", "${confLive.toStringAsFixed(1)}%", isLive ? Colors.greenAccent : Colors.white70),
                  Container(width: 1, height: 30, color: Colors.white12),
                  _buildStatItem("Probabilitas Spoof", "${confSpoof.toStringAsFixed(1)}%", !isLive ? Colors.redAccent : Colors.white70),
                  Container(width: 1, height: 30, color: Colors.white12),
                  _buildStatItem("Latensi GPU", "${latency.toStringAsFixed(0)} ms", Colors.blueAccent),
                ],
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildStatItem(String label, String value, Color color) {
    return Column(
      children: [
        Text(label, style: const TextStyle(fontSize: 11, color: Colors.white54)),
        const SizedBox(height: 3),
        Text(value, style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: color)),
      ],
    );
  }
}
