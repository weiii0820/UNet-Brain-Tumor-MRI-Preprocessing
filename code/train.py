from data_process import load_images_and_masks
from metrics import DiceCoefficient, SafeMeanIoU, combined_bce_tversky_loss

from build import build
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt



gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU 記憶體設定為逐步增長模式")
    except RuntimeError as e:
        print("⚠️ GPU 記憶體設定失敗：", e)
mixed_precision.set_global_policy('mixed_float16')


def augment(image, mask):
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.rot90(image)
        mask = tf.image.rot90(mask)
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_brightness(image, max_delta=0.2)  # 可調整亮度變化幅度
    if tf.random.uniform(()) > 0.5:
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # 對比調整

    return image, mask

def show_pred_results(model, x_val, y_val, start_index=0, num_images=5, threshold=0.3):
    """
    顯示多張模型預測結果，包括原圖、真實 mask 與預測 mask。

    Parameters:
        model: 已訓練模型
        x_val: 驗證影像資料 (numpy array)
        y_val: 驗證標註資料 (numpy array)
        start_index: 從第幾張影像開始顯示
        num_images: 顯示幾張
        threshold: 閾值，將 sigmoid 預測轉為二值 mask
    """
    end_index = min(start_index + num_images, len(x_val))
    preds = model.predict(x_val[start_index:end_index])

    for i in range(end_index - start_index):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.title("影像")
        plt.imshow(x_val[start_index + i].squeeze(), cmap='gray')
        
        plt.subplot(1, 3, 2)
        plt.title("真實標記")
        plt.imshow(y_val[start_index + i].squeeze(), cmap='gray')
        
        plt.subplot(1, 3, 3)
        plt.title("模型預測")
        plt.imshow((preds[i].squeeze() > threshold), cmap='gray')
        
        plt.suptitle(f"第 {start_index + i} 張", fontsize=14)
        plt.tight_layout()
        plt.show()

def plot_training_history(history):
    metrics = ['loss', 'accuracy', 'dice_coef', 'safe_mean_iou']
    
    for metric in metrics:
        plt.figure(figsize=(8, 5))
        plt.plot(history.history[metric], label=f"train_{metric}")
        plt.plot(history.history[f"val_{metric}"], label=f"val_{metric}")
        plt.title(f"{metric} during training")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


IMG_SIZE = (256, 256)
BATCH_SIZE = 4
EPOCHS = 50
IMAGE_DIR = "dataset/train/imgs"
MASK_DIR = "dataset/train/masks"
BEST_MODEL_PATH = "best_unet_model.h5"


x, y = load_images_and_masks(IMG_SIZE, IMAGE_DIR, MASK_DIR)
np.save('images.npy', x)
np.save('masks.npy', y)

# x = np.load('images.npy')
# y = np.load('masks.npy')

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.15, random_state=42)
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(augment).batch(BATCH_SIZE).repeat()

model = build(input_shape=(256, 256, 1))
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=combined_bce_tversky_loss,
    metrics=['accuracy', DiceCoefficient(), SafeMeanIoU(threshold=0.3)]
)

checkpoint = ModelCheckpoint(BEST_MODEL_PATH, monitor='val_loss', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=1)

history = model.fit(
        train_dataset,
        steps_per_epoch=len(x_train) // BATCH_SIZE,
        validation_data=(x_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=[checkpoint, earlystop, reduce_lr])

show_pred_results(model, x_val, y_val, start_index=0, num_images=5)
plot_training_history(history)

model.load_weights(BEST_MODEL_PATH)
model.save("unet_final_model.h5")

