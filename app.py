import os
import io
import base64
import random
import numpy as np
from PIL import Image
from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

# MNIST 테스트 데이터 전역 변수로 로드
mnist_test_data = None
mnist_test_labels = None

def load_mnist_test_data():
    """MNIST 테스트 데이터를 로드합니다"""
    global mnist_test_data, mnist_test_labels
    if mnist_test_data is None:
        try:
            (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            mnist_test_data = x_test
            mnist_test_labels = y_test
            print(f"MNIST 테스트 데이터 로드 완료: {len(x_test)}개 이미지")
        except Exception as e:
            print(f"MNIST 테스트 데이터 로드 실패: {e}")
    return mnist_test_data, mnist_test_labels

def create_compatible_model():
    """호환 가능한 MNIST CNN 모델 생성"""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam', 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    return model

# 모델 로드
try:
    # 기존 모델 로드 시도
    model = tf.keras.models.load_model('mnist_cnn_model.keras')
    print("기존 모델이 성공적으로 로드되었습니다.")
except Exception as e:
    print(f"기존 모델 로드 실패. 새 모델을 생성합니다.")
    print("원인:", str(e)[:100] + "...")
    
    # 호환 가능한 새 모델 생성
    model = create_compatible_model()
    
    # 빠른 MNIST 훈련으로 기본 가중치 설정
    print("기본 MNIST 데이터로 간단한 훈련을 수행합니다...")
    try:
        (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
        x_train = x_train.astype('float32') / 255.0  # 1000개 샘플만 사용
        x_train = x_train.reshape(-1, 28, 28, 1)
        y_train = tf.keras.utils.to_categorical(y_train, 10)
        
        model.fit(x_train, y_train, epochs=1, verbose=0)
        print("기본 훈련 완료. 모델이 준비되었습니다.")
        
        # 새 모델 저장
        model.save('mnist_cnn_model.keras')
        print("호환 모델을 저장했습니다: mnist_cnn_model.keras")
    except Exception as train_error:
        print(f"훈련 실패, 임의 가중치로 진행: {train_error}")

if model is not None:
    print("모델 준비 완료!")
    print("입력 형태:", model.input_shape)
    print("출력 형태:", model.output_shape)
else:
    print("모델 로드/생성에 실패했습니다.")

def preprocess_image(image):
    """이미지를 MNIST 모델 입력 형태로 전처리"""
    # 컬러 이미지를 그레이스케일로 변환
    if image.mode == 'RGBA':
        # RGBA를 RGB로 변환 (흰색 배경에 합성)
        background = Image.new('RGB', image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[-1])
        image = background
    
    # RGB나 다른 컬러 형식을 그레이스케일로 변환
    if image.mode != 'L':
        image = image.convert('L')  # 'L' 모드는 그레이스케일
    
    # 28x28로 리사이즈
    image = image.resize((28, 28), Image.Resampling.LANCZOS)
    
    # 배경을 흰색으로, 텍스트를 검은색으로 (MNIST 형태)
    image_array = np.array(image)
    # 픽셀 값 반전 (255-픽셀값)하여 흰 배경을 검은 배경으로, 검은 글씨를 흰 글씨로
    image_array = 255 - image_array
    
    # 정규화 (0-1 범위)
    image_array = image_array.astype('float32') / 255.0
    
    # 차원 추가 (batch_size, height, width, channels)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = np.expand_dims(image_array, axis=-1)
    
    return image_array

def image_to_base64(image):
    """PIL Image를 base64 문자열로 변환"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': '모델이 로드되지 않았습니다.'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': '파일이 업로드되지 않았습니다.'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '파일이 선택되지 않았습니다.'}), 400
    
    try:
        # 이미지 읽기
        image = Image.open(file.stream)
        original_size = image.size
        
        # 전처리
        processed_image_array = preprocess_image(image)
        
        # 예측
        prediction = model.predict(processed_image_array)
        
        # TOP-5 결과 계산
        prediction_probs = prediction[0]
        top5_indices = np.argsort(prediction_probs)[::-1][:5]
        
        top1_results = [
            {'class': int(top5_indices[0]), 'confidence': float(prediction_probs[top5_indices[0]])}
        ]
        
        top5_results = [
            {'class': int(idx), 'confidence': float(prediction_probs[idx])} 
            for idx in top5_indices
        ]
        
        # 전처리된 이미지를 시각화를 위해 PIL Image로 변환
        processed_display = processed_image_array[0, :, :, 0] * 255
        processed_display = processed_display.astype(np.uint8)
        processed_image_pil = Image.fromarray(processed_display, mode='L')
        
        # 결과 반환
        result = {
            'top1_results': top1_results,
            'top5_results': top5_results,
            'original_size': original_size,
            'processed_size': (28, 28),
            'original_image': image_to_base64(image.resize((200, 200), Image.Resampling.LANCZOS)),
            'processed_image': image_to_base64(processed_image_pil.resize((200, 200), Image.Resampling.NEAREST))
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'이미지 처리 중 오류가 발생했습니다: {str(e)}'}), 500

@app.route('/load_mnist', methods=['POST'])
def load_mnist():
    """랜덤 MNIST 테스트 이미지를 로드합니다"""
    try:
        # MNIST 테스트 데이터 로드
        test_data, test_labels = load_mnist_test_data()
        
        if test_data is None:
            return jsonify({'error': 'MNIST 테스트 데이터를 로드할 수 없습니다.'}), 500
        
        # 랜덤 인덱스 선택
        random_idx = random.randint(0, len(test_data) - 1)
        
        # 선택된 이미지와 라벨
        selected_image = test_data[random_idx]
        true_label = int(test_labels[random_idx])
        
        # 이미지를 PIL Image로 변환 (28x28 그레이스케일)
        mnist_image = Image.fromarray(selected_image, mode='L')
        
        # 56x56으로 리사이즈 (원본 이미지 창용)
        resized_image = mnist_image.resize((56, 56), Image.Resampling.NEAREST)
        
        # 모델 예측을 위한 전처리
        image_array = selected_image.astype('float32') / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        image_array = np.expand_dims(image_array, axis=-1)
        
        # 예측 수행
        if model is not None:
            prediction = model.predict(image_array)
            prediction_probs = prediction[0]
            top5_indices = np.argsort(prediction_probs)[::-1][:5]
            
            top1_results = [
                {'class': int(top5_indices[0]), 'confidence': float(prediction_probs[top5_indices[0]])}
            ]
            
            top5_results = [
                {'class': int(idx), 'confidence': float(prediction_probs[idx])} 
                for idx in top5_indices
            ]
        else:
            top1_results = [{'class': -1, 'confidence': 0.0}]
            top5_results = [{'class': -1, 'confidence': 0.0}]
        
        # 전처리된 이미지 (28x28) - 표시용으로 200x200으로 리사이즈
        processed_display = Image.fromarray(selected_image, mode='L')
        processed_display_resized = processed_display.resize((200, 200), Image.Resampling.NEAREST)
        
        result = {
            'top1_results': top1_results,
            'top5_results': top5_results,
            'original_size': (56, 56),  # 표시될 원본 이미지 크기
            'processed_size': (28, 28),
            'original_image': image_to_base64(resized_image),  # 56x56 크기
            'processed_image': image_to_base64(processed_display_resized),  # 200x200 표시용
            'true_label': true_label,
            'mnist_index': random_idx
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': f'MNIST 이미지 로드 중 오류가 발생했습니다: {str(e)}'}), 500

if __name__ == '__main__':
    # 앱 시작 시 MNIST 데이터 미리 로드
    print("MNIST 테스트 데이터를 로드하는 중...")
    load_mnist_test_data()
    
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=True)
