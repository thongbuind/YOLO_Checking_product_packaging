AssemblyCheck-M1-Hybrid/                  ← Tên project (mở bằng VSCode)
│
├── python/                               ← Toàn bộ phần Python (chỉ đọc cam + YOLO)
│   ├── main.py                           ← Main loop (bạn đang có)
│   ├── camera_reader.py                  ← 4 thread đọc camera (hoặc dùng hiện tại)
│   ├── yolo_batch_processor.py           ← Ultralytics YOLOv8-OBB + Torch MPS
│   ├── shared_memory.py                  ← Zero-copy ring buffer (mmap)
│   ├── config_cams.yaml                  ← cam_id, expected_slots, delay_duration...
│   └── requirements.txt
│
├── swift/                                ← TOÀN BỘ XỬ LÝ + VẼ (Swift Package → compile thành 1 file .dylib)
│   ├── VisionEngine/                     ← Swift Package (có thể mở bằng Xcode hoặc VSCode)
│   │   ├── Package.swift                 ← Manifest
│   │   ├── Sources/VisionEngine/
│   │   │   ├── VisionEngine.swift        ← Public class (Python sẽ gọi)
│   │   │   ├── CameraProcessor.swift     ← 1 instance cho mỗi cam (state machine 100% như Python cũ)
│   │   │   ├── SlotMapper.swift          ← Hàm slot_position() port từ Python
│   │   │   ├── GeometryEngine.swift      ← IoU OBB, contains, collision → dùng Accelerate SIMD
│   │   │   ├── MetalRenderer.swift       ← Metal device + command queue
│   │   │   ├── Renderer.swift            ← Vẽ polyline + text siêu nhanh
│   │   │   ├── Shaders.metal             ← 3 shader: polyline, filled_polygon, sdf_text
│   │   │   ├── ShaderTypes.h             ← Bridge header C
│   │   │   └── FontSDF.png               ← Font texture (đã bake sẵn)
│   │   │
│   │   └── Resources/
│   │       └── FontSDF.png                   ← Signed Distance Field font
│   │
│   ├── build.sh                          ← 1 click compile Swift → libVisionEngine.dylib
│   └── libVisionEngine.dylib                 ← File duy nhất Python sẽ load (sau khi build)
│
├── bridge/                               ← Giao tiếp Python ↔ Swift (siêu nhẹ)
│   └── swift_bridge.py                   ← Dùng ctypes load .dylib + định nghĩa struct
│
├── shared/                               ← Dữ liệu trao đổi (zero-copy)
│   └── ringbuffer.bin                    ← File mmap 200MB (tự tạo khi chạy lần đầu)
│
├── scripts/
│   ├── start.sh                          ← Chạy toàn bộ: Python + Swift engine
│   ├── build_swift.sh                    ← Compile Swift (dùng trong VSCode task)
│   └── deploy/                           ← Folder để copy lên Mac Mini nhà máy
│       ├── AssemblyCheck.app/            ← (tùy chọn) hoặc chỉ chạy script
│       └── run.sh
│
├── README.md                             ← Hướng dẫn từng bước (có mình viết sẵn)
└── .vscode/
    ├── tasks.json                        ← Build Swift + Run bằng 1 phím tắt
    └── launch.json                       ← Debug Python (có thể attach vào Swift sau)
    