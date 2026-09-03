"""
test_sunglasses_pretrained.py
------------------------------
Standalone webcam sunglasses test using the pretrained `glasses-detector`

Install:
    pip install glasses-detector
Press 'q' to quit.
"""

import cv2
from glasses_detector import GlassesClassifier

classifier = GlassesClassifier(size="small", kind="sunglasses")


def classify_frame(frame_rgb) -> bool:
    """
    Tries the library's likely single-image inference call. Different
    versions of glasses-detector may expose this slightly differently, so
    this tries a couple of common conventions before giving up.
    """
    if hasattr(classifier, "predict"):
        result = classifier.predict(frame_rgb, format="bool")
        return bool(result)

    # Fallback: some versions make the classifier itself callable.
    result = classifier(frame_rgb)
    return bool(result)


def main():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            sunglasses_on = classify_frame(rgb)
        except Exception as e:
            print(f"[ERROR] Classifier call failed: {e}")
            print("Run: python -c \"from glasses_detector import GlassesClassifier; "
                  "c = GlassesClassifier(kind='sunglasses'); print([m for m in dir(c) if not m.startswith('_')])\"")
            print("to find the correct method name, then update classify_frame().")
            break

        status = "SUNGLASSES ON" if sunglasses_on else "BARE EYES"
        color = (0, 0, 255) if sunglasses_on else (0, 255, 0)
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.imshow("Sunglasses Test (pretrained)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()