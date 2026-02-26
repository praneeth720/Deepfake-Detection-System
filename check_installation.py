print("🔍 Checking Python packages...\n")

packages = [
    "torch",
    "torchvision",
    "transformers",
    "accelerate",
    "cv2",
    "numpy",
    "PIL",
    "flask",
    "pytesseract"
]

for pkg in packages:
    try:
        __import__(pkg)
        print(f"✅ {pkg} installed")
    except ImportError:
        print(f"❌ {pkg} NOT installed")

print("\n🔍 Checking Tesseract OCR engine...\n")

try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    print("Tesseract version:")
    print(pytesseract.get_tesseract_version())
except Exception as e:
    print("⚠ OCR installed but version check skipped")
    print(e)