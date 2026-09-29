"""
face_identifier.py
------------------
Face identification module using MobileFaceNet INT8 TFLite model and
uniface SCRFD landmark alignment, matching against employee dataset embeddings.

Links to the `employee` folder and `models/mobilefacenet_int8.tflite`.
"""

import os
import sys
import time
from typing import Optional, Tuple, Dict, List

import cv2
import numpy as np

# Try importing TFLite Interpreter from available packages
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite import Interpreter
        except ImportError:
            raise ImportError(
                "Could not import Interpreter. Please install `ai-edge-litert`, "
                "`tflite-runtime`, or `tensorflow`."
            )

# Standard 5-point face template (MobileFaceNet/ArcFace style)
REFERENCE_LANDMARKS = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float32,
)

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models",
    "mobilefacenet_int8.tflite",
)

DEFAULT_EMPLOYEE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "employee",
)


def align_face(image: np.ndarray, landmarks: np.ndarray) -> Optional[np.ndarray]:
    """
    Align a face image to 112x112 using 5 2D landmarks.
    """
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.shape != (5, 2):
        return None

    transform, _ = cv2.estimateAffinePartial2D(
        landmarks, REFERENCE_LANDMARKS, method=cv2.LMEDS
    )

    if transform is None:
        return None

    aligned = cv2.warpAffine(
        image,
        transform,
        (112, 112),
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return aligned


class FaceIdentifier:
    """
    MobileFaceNet INT8 face identification manager.
    Loads model, builds employee embedding database using uniface SCRFD analyzer,
    and matches live input faces against the database.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        employee_dir: str = DEFAULT_EMPLOYEE_DIR,
        threshold: float = 0.50,
    ):
        if not os.path.exists(model_path):
            fallback_model = "/home/vivek/Music/face/mobile_net/mobilefacenet_int8.tflite"
            if os.path.exists(fallback_model):
                model_path = fallback_model
            else:
                raise FileNotFoundError(f"MobileFaceNet model not found at {model_path}")

        if not os.path.exists(employee_dir):
            fallback_emp = "/home/vivek/Music/face/employee"
            if os.path.exists(fallback_emp):
                employee_dir = fallback_emp

        self.model_path = model_path
        self.employee_dir = employee_dir
        self.threshold = threshold

        print(f"[INFO] Initializing MobileFaceNet INT8: {self.model_path}")
        self.interpreter = Interpreter(model_path=self.model_path)
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]

        self.input_scale, self.input_zero = self.input_details["quantization"]
        self.output_scale, self.output_zero = self.output_details["quantization"]

        # Initialize uniface FaceAnalyzer for landmark alignment
        self.analyzer = None
        try:
            from uniface import FaceAnalyzer
            print("[INFO] Initializing uniface FaceAnalyzer (SCRFD)...")
            self.analyzer = FaceAnalyzer()
        except Exception as e:
            print(f"[WARN] Could not initialize uniface FaceAnalyzer: {e}")

        self.employee_embeddings: Optional[np.ndarray] = None
        self.employee_ids: List[str] = []
        self.employee_names: List[str] = []

    def get_embedding(self, face_image: np.ndarray) -> Optional[np.ndarray]:
        """
        Computes MobileFaceNet INT8 embedding vector for a 112x112 aligned face.
        """
        if face_image is None or face_image.size == 0:
            return None

        if face_image.shape[:2] != (112, 112):
            face_image = cv2.resize(face_image, (112, 112))

        # OpenCV = BGR -> Model expects RGB
        rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)

        # Convert to float [0, 1]
        image = rgb.astype(np.float32) / 255.0

        # Quantize according to model parameters
        image = np.round(image / self.input_scale) + self.input_zero
        image = np.clip(image, -128, 127).astype(np.int8)
        image = np.expand_dims(image, axis=0)

        self.interpreter.set_tensor(self.input_details["index"], image)
        self.interpreter.invoke()

        output = self.interpreter.get_tensor(self.output_details["index"])

        # Dequantize & remove batch dimension
        embedding = output[0].astype(np.float32)
        embedding = (embedding - self.output_zero) * self.output_scale

        # L2 normalization
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return None

        return embedding / norm

    def load_employee_database(self) -> int:
        """
        Builds employee database from images in self.employee_dir.
        Returns count of valid employee embeddings created.
        """
        print("=" * 70)
        print("Building Employee Database")
        print("=" * 70)

        if not os.path.exists(self.employee_dir):
            print(f"[ERROR] Employee directory not found: {self.employee_dir}")
            return 0

        files = sorted(
            f for f in os.listdir(self.employee_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )

        print(f"Employee images found: {len(files)}")

        embeddings_list = []
        ids_list = []
        names_list = []
        failed = 0

        for filename in files:
            path = os.path.join(self.employee_dir, filename)
            image = cv2.imread(path)
            if image is None:
                print(f"[ERROR] Cannot read: {filename}")
                failed += 1
                continue

            aligned = None
            if self.analyzer is not None:
                try:
                    faces = self.analyzer.analyze(image)
                    if faces:
                        largest_face = max(
                            faces,
                            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                        )
                        if largest_face.landmarks is not None:
                            aligned = align_face(image, largest_face.landmarks)
                except Exception:
                    pass

            if aligned is None:
                aligned = cv2.resize(image, (112, 112))

            embedding = self.get_embedding(aligned)
            if embedding is None:
                print(f"[EMBEDDING FAILED] {filename}")
                failed += 1
                continue

            basename = os.path.splitext(filename)[0]
            if "_" in basename:
                emp_id, emp_name = basename.split("_", 1)
            else:
                emp_id = basename
                emp_name = basename

            embeddings_list.append(embedding)
            ids_list.append(emp_id)
            names_list.append(emp_name)
            print(f"[OK] {emp_id:<8} {emp_name}")

        if embeddings_list:
            self.employee_embeddings = np.asarray(embeddings_list, dtype=np.float32)
            self.employee_ids = ids_list
            self.employee_names = names_list

        print("=" * 70)
        print(f"DATABASE READY: {len(self.employee_names)} valid employees loaded ({failed} failed)")
        print("=" * 70)

        return len(self.employee_names)

    def identify(self, aligned_face_image: np.ndarray) -> Tuple[str, str, str, float]:
        """
        Identifies a face from an aligned 112x112 face crop image.
        Returns: (status, employee_id, employee_name, similarity_score)
        """
        if self.employee_embeddings is None or len(self.employee_embeddings) == 0:
            return "UNKNOWN", "UNKNOWN", "Unknown", 0.0

        embedding = self.get_embedding(aligned_face_image)
        if embedding is None:
            return "UNKNOWN", "UNKNOWN", "Unknown", 0.0

        scores = np.dot(self.employee_embeddings, embedding)
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        best_id = self.employee_ids[best_idx]
        best_name = self.employee_names[best_idx]

        if best_score >= self.threshold:
            return "KNOWN", best_id, best_name, best_score
        else:
            return "UNKNOWN", best_id, best_name, best_score

    def identify_frame(
        self, frame: np.ndarray
    ) -> Tuple[str, str, str, float, Optional[Tuple[int, int, int, int]]]:
        """
        Analyzes a camera frame using Uniface SCRFD face detector, aligns the face,
        computes MobileFaceNet embedding, and matches against employee database.
        
        Returns: (status, employee_id, employee_name, similarity_score, (x1, y1, x2, y2))
        """
        if self.employee_embeddings is None or len(self.employee_embeddings) == 0:
            return "UNKNOWN", "UNKNOWN", "Unknown", 0.0, None

        if self.analyzer is not None:
            try:
                faces = self.analyzer.analyze(frame)
                if faces:
                    face = max(
                        faces,
                        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                    )
                    if face.landmarks is not None:
                        aligned = align_face(frame, face.landmarks)
                        if aligned is not None:
                            status, emp_id, emp_name, score = self.identify(aligned)
                            bbox = (
                                int(face.bbox[0]),
                                int(face.bbox[1]),
                                int(face.bbox[2]),
                                int(face.bbox[3]),
                            )
                            return status, emp_id, emp_name, score, bbox
            except Exception as e:
                pass

        return "UNKNOWN", "UNKNOWN", "Unknown", 0.0, None

    def draw_face_box(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
        driver_status: str,
        driver_id: str,
        driver_name: str,
        driver_score: float,
    ) -> None:
        """
        Draws the face bounding box, status (KNOWN/UNKNOWN), person name, and similarity score directly on the frame.
        """
        if not bbox:
            return

        x1, y1, x2, y2 = bbox

        if driver_status == "KNOWN":
            color = (0, 255, 0)  # Green in BGR
            display_name = f"{driver_id} - {driver_name}"
        else:
            color = (0, 0, 255)  # Red in BGR
            display_name = "UNKNOWN"

        # Draw Face Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Draw Person Name / ID above the box
        cv2.putText(
            frame,
            display_name,
            (x1, max(30, y1 - 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2,
        )

        # Draw Status (KNOWN/UNKNOWN) and Score
        score_text = f"{driver_status} {driver_score:.3f}"
        cv2.putText(
            frame,
            score_text,
            (x1, max(55, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


if __name__ == "__main__":
    identifier = FaceIdentifier()
    identifier.load_employee_database()
