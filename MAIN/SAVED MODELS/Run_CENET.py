from glob import glob
from tensorflow.keras.metrics import Precision, Recall, MeanIoU
from tensorflow.keras.optimizers import Adam, Nadam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from CE_NET import cenet
from CE_NET.metrics import dice_coef, dice_loss
from CE_NET.data_generator import *
import numpy as np
import read
import re
import os


def mask_to_3d(mask):
    mask = np.squeeze(mask)
    mask = [mask, mask, mask]
    mask = np.transpose(mask, (1, 2, 0))
    return mask
if __name__ == "__main__":
    ## Path
    file_path = "CE_NET/files/"
    model_path = "CE_NET/files/cenet.h5"

    ## Create files folder
    try:
        os.mkdir("CE_NET/files")
    except:
        pass

    ## Training
    train_image_paths = read.image('Dataset/test/*/*')
    train_mask_paths = read.image('Processed/gt/*')

    ## Validation
    valid_image_paths = read.image('Dataset/test/*/*')
    valid_mask_paths = read.image('Processed/gt/*')


    ## Parameters
    image_size = 256
    batch_size = 8
    lr = 1e-4
    epochs = 500

    train_steps = len(train_image_paths)//batch_size
    valid_steps = len(valid_image_paths)//batch_size

    ## Generator
    train_gen = DataGen(image_size, train_image_paths, train_mask_paths, batch_size=batch_size)
    valid_gen = DataGen(image_size, valid_image_paths, valid_mask_paths, batch_size=batch_size)

    #CE_NET
    arch = cenet.CE_Net_.forward(image_size)
    model = arch.build_model()


    optimizer = Nadam(lr)
    metrics = [Recall(), Precision(), dice_coef, MeanIoU(num_classes=2)]
    model.compile(loss=dice_loss, optimizer=optimizer, metrics=metrics)

    csv_logger = CSVLogger(f"{file_path}cenet{batch_size}.csv", append=False)
    checkpoint = ModelCheckpoint(model_path, verbose=1, save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=False)
    callbacks = [csv_logger, checkpoint, reduce_lr, early_stopping]

    model.fit_generator(train_gen,
            validation_data=valid_gen,
            steps_per_epoch=train_steps,
            validation_steps=valid_steps,
            epochs=epochs,
            callbacks=callbacks)


    #--------testing-------------------
    from tqdm import tqdm

    save_path = "Processed/Segmented_Image"

    test_mask_paths = read.image('Processed/gt/*')

    test_image_paths = read.image('Dataset/test/*/*')

    ## Generating the result
    for i, path in tqdm(enumerate(test_image_paths), total=len(test_image_paths)):
        image = parse_image(test_image_paths[i], image_size)
        mask = parse_mask(test_mask_paths[i], image_size)


        predict_mask = model.predict(np.expand_dims(image, axis=0))[0]
        predict_mask = (predict_mask > 0.5) * 255.0

        sep_line = np.ones((image_size, 10, 3)) * 255

        mask = mask_to_3d(mask)
        predict_mask = mask_to_3d(predict_mask)

        all_images = [image * 255, sep_line, mask * 255, sep_line, predict_mask]
        cv2.imwrite(f"{save_path}/{i}.jpg", predict_mask)



    print("Test image generation complete")
