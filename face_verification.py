from flask import Flask, jsonify, request
import requests
import face_recognition
import numpy as np
import io
import time
import traceback

app = Flask(__name__)

def download_image(url, max_retries=3, delay=1):
    """Tải ảnh từ URL với tối đa 3 lần thử lại."""
    for attempt in range(max_retries):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.content
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            app.logger.warning(f"Attempt {attempt + 1} failed for URL {url}: {str(e)}")
            if attempt < max_retries - 1:
                time.sleep(delay)  # Đợi trước khi thử lại
                continue
            app.logger.error(f"Failed to download image from {url} after {max_retries} attempts")
            raise Exception(f"Failed to download image after {max_retries} attempts: {str(e)}")

@app.route('/verify', methods=['POST'])
def verify_face():
    try:
        # Nhận dữ liệu từ Node.js với tên trường đúng
        reference_image_url = request.json.get('referenceImageUrl')
        captured_image_url = request.json.get('capturedImageUrl')

        if not reference_image_url or not captured_image_url:
            app.logger.error("Missing referenceImageUrl or capturedImageUrl")
            return jsonify({'error': 'Missing referenceImageUrl or capturedImageUrl'}), 400

        # Tải ảnh với cơ chế thử lại
        reference_image_data = download_image(reference_image_url)
        captured_image_data = download_image(captured_image_url)

        # Chuyển đổi dữ liệu ảnh thành định dạng numpy array
        reference_image = face_recognition.load_image_file(io.BytesIO(reference_image_data))
        captured_image = face_recognition.load_image_file(io.BytesIO(captured_image_data))

        # Lấy encodings của khuôn mặt
        try:
            reference_encodings = face_recognition.face_encodings(reference_image)
            captured_encodings = face_recognition.face_encodings(captured_image)
        except Exception as e:
            app.logger.error(f"Could not process face features: {str(e)}")
            return jsonify({'match': False, 'error': f'Could not process face features: {str(e)}'}), 200

        app.logger.info(f"Number of reference face encodings: {len(reference_encodings)}")
        app.logger.info(f"Number of captured face encodings: {len(captured_encodings)}")

        # Kiểm tra số khuôn mặt
        if len(reference_encodings) == 0:
            app.logger.warning("No face detected in reference image")
            return jsonify({'match': False, 'error': 'No face detected in reference image'}), 200
        if len(captured_encodings) == 0:
            app.logger.warning("No face detected in captured image")
            return jsonify({'match': False, 'error': 'No face detected in captured image'}), 200
        if len(reference_encodings) > 1:
            app.logger.warning(f"Multiple faces ({len(reference_encodings)}) detected in reference image. Using the first one.")
        if len(captured_encodings) > 1:
            app.logger.warning(f"Multiple faces ({len(captured_encodings)}) detected in captured image. Using the first one.")

        # So sánh khuôn mặt
        app.logger.info("Comparing faces with tolerance=0.4...")
        ref_encoding = reference_encodings[0]
        cap_encoding = captured_encodings[0]

        distance = face_recognition.face_distance([ref_encoding], cap_encoding)[0]
        is_match = face_recognition.compare_faces([ref_encoding], cap_encoding, tolerance=0.4)[0]

        app.logger.info(f"Face comparison result: match={is_match}, distance={distance:.4f}")

        return jsonify({'match': bool(is_match), 'distance': float(distance)})

    except Exception as e:
        app.logger.error(f"Unexpected error in face verification: {str(e)}", exc_info=True)
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500

if __name__ == '__main__':
    print("Starting Flask server for face verification...")
    app.logger.info("Flask server for face verification starting...")
    app.run(host='0.0.0.0', port=5000, debug=False)