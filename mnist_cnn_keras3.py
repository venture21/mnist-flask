import keras
import numpy as np
import matplotlib.pyplot as plt

def create_cnn_model():
    """CNN 모델 생성"""
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Conv2D(64, (3,3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(10, activation='softmax')
    ])
    return model

def load_and_preprocess_data():
    """MNIST 데이터 로드 및 전처리"""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # 정규화 (0-1 범위로 변환)
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 차원 추가 (28, 28) -> (28, 28, 1)
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    
    # 원-핫 인코딩
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def train_model():
    """모델 학습"""
    # 데이터 로드
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # 모델 생성
    model = create_cnn_model()
    
    # 모델 컴파일
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 모델 구조 출력
    print("모델 구조:")
    model.summary()
    
    # 콜백 설정
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=2)
    ]
    
    # 모델 학습
    print("\n모델 학습 시작...")
    history = model.fit(
        x_train, y_train,
        epochs=20,
        batch_size=128,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    # 최종 평가
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n최종 테스트 정확도: {test_acc:.4f}")
    
    # 모델 저장
    model.save('mnist_cnn_model.keras')
    print("모델이 'mnist_cnn_model.keras'로 저장되었습니다.")
    
    return model, history

def plot_training_history(history):
    """학습 과정 시각화"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 정확도 플롯
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # 손실 플롯
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

if __name__ == "__main__":
    # 모델 학습 실행
    model, history = train_model()
    
    # 학습 과정 시각화
    plot_training_history(history)