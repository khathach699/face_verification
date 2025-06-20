# File: face_verification.py

from flask import Flask, request, jsonify
import face_recognition
import requests
from io import BytesIO
import logging

app = Flask(__name__)
# Cấu hình logging để ghi ra file và hiển thị trên console
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/verify', methods=['POST'])
def verify_face():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Invalid JSON body'}), 400

        captured_url = data.get('capturedImageUrl')
        reference_url = data.get('referenceImageUrl')

        if not captured_url or not reference_url:
            app.logger.warning('Missing image URLs in request.')
            return jsonify({'error': 'Missing image URLs'}), 400

        app.logger.info(f"Attempting to download reference image: {reference_url}")
        reference_response = requests.get(reference_url, timeout=20) # Tăng timeout lên 20 giây
        reference_response.raise_for_status() # Ném lỗi nếu status code là 4xx hoặc 5xx

        app.logger.info(f"Attempting to download captured image: {captured_url}")
        captured_response = requests.get(captured_url, timeout=20) # Tăng timeout lên 20 giây
        captured_response.raise_for_status() # Ném lỗi

        app.logger.info("Images downloaded successfully. Loading images for recognition.")
        
        captured_img = face_recognition.load_image_file(BytesIO(captured_response.content))
        reference_img = face_recognition.load_image_file(BytesIO(reference_response.content))

        app.logger.info("Getting face encodings.")
        # Lấy encoding của khuôn mặt đầu tiên tìm thấy
        reference_encodings = face_recognition.face_encodings(reference_img)
        if not reference_encodings:
            app.logger.warning("No face detected in the reference image.")
            return jsonify({'match': False, 'error': 'No face detected in reference image'}), 200

        captured_encodings = face_recognition.face_encodings(captured_img)
        if not captured_encodings:
            app.logger.warning("No face detected in the captured image.")
            return jsonify({'match': False, 'error': 'No face detected in captured image'}), 200

        app.logger.info("Comparing faces.")
        # So sánh khuôn mặt
        is_match_np = face_recognition.compare_faces([reference_encodings[0]], captured_encodings[0])[0]
        
        # --- DÒNG SỬA LỖI QUAN TRỌNG ---
        # Chuyển đổi numpy.bool_ thành bool của Python để jsonify hoạt động chắc chắn
        is_match_python = bool(is_match_np)
        
        app.logger.info(f"Verification result: {is_match_python}")
        return jsonify({'match': is_match_python})

    except requests.exceptions.Timeout:
        logging.error("Request to download image timed out.")
        return jsonify({'error': 'Failed to download image: Connection timed out.'}), 500
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading image: {str(e)}")
        return jsonify({'error': f"Failed to download image: {str(e)}"}), 500
    except Exception as e:
        logging.error(f"An unexpected error occurred in face verification: {str(e)}")
        # Trả về lỗi chi tiết để debug, nhưng trong production nên che giấu
        return jsonify({'error': 'An internal server error occurred', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)