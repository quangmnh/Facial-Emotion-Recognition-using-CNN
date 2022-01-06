from imgaug import augmenters as iaa
import cv2
import numpy as np
import glob2

# Loading DNN model
modelFile = 'res10_300x300_ssd_iter_140000.caffemodel'
configFile = 'deploy.prototxt.txt'
net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
conf_threshold = 0.5

# Model for generating augmentation data
def generate(img):

    seq_0 = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0.0, 1.0))
    ])

    seq_1 = iaa.Sequential([
        iaa.AverageBlur(k=(1, 3))
    ])

    seq_2 = iaa.Sequential([
        # iaa.BilateralBlur(d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250))
        iaa.Dropout(0.01)
    ])

    seq_3 = iaa.Sequential([
        iaa.Add((-40, 40)),
        iaa.GaussianBlur(sigma=(0.0, 1.0))
    ])

    seq_4 = iaa.Sequential([
        iaa.Dropout(0.01),
        iaa.Add((-40, 40))
    ])

    seq_5 = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0.0, 0.5)),
        iaa.AveragePooling(kernel_size=1)
    ])

    seq_6 = iaa.Sequential([
        iaa.pillike.EnhanceColor(factor=0.5)
    ])

    seq_7 = iaa.Sequential([
        iaa.pillike.EnhanceColor(factor=0.5),
        iaa.GaussianBlur(sigma=(0.0, 1.0))
    ])

    seq_8 = iaa.Sequential([
        iaa.Add((-40, 40)),
        iaa.AveragePooling(kernel_size=1)
    ])

    seq_9 = iaa.Sequential([
        iaa.ImpulseNoise(0.01),
        iaa.Flipud(0.5)
    ])

    seq_10 = iaa.Sequential([
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
        # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))
    ])

    seq_11 = iaa.Sequential([
        iaa.Add(value=-45),
        iaa.GammaContrast(gamma=1.44),
        iaa.pillike.EnhanceSharpness(factor=0.15),
        # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0))
    ])

    seq_12 = iaa.Sequential([
        iaa.GammaContrast(gamma=1.44),
        iaa.pillike.EnhanceSharpness(factor=0.15),
    ])

    seq_13 = iaa.Sequential([
        iaa.GaussianBlur(sigma=(0.0, 0.1)),
        iaa.LogContrast(gain=0.75)
    ])

    seq_14 = iaa.Sequential([
        iaa.contrast.LogContrast(gain=0.73),
        iaa.GaussianBlur(sigma=(0.0, 1.0))
    ])

    seq_15 = iaa.Sequential([
        iaa.GammaContrast(gamma=1.44, per_channel=True),
        iaa.GaussianBlur(sigma=(0.0, 0.1))
    ])

    seq_16 = iaa.Sequential([
        iaa.LinearContrast(alpha=1.10),
        iaa.Add(value=-45)
    ])

    image_aug = [
        img,
        seq_0(image=img),
        seq_1(image=img),
        seq_2(image=img),
        seq_3(image=img),
        seq_4(image=img),
        seq_5(image=img),
        seq_6(image=img),
        seq_7(image=img),
        seq_8(image=img),
        seq_9(image=img),
        seq_10(image=img),
        seq_11(image=img),
        seq_12(image=img),
        seq_13(image=img),
        seq_14(image=img),
        seq_15(image=img),
        seq_16(image=img)
    ]

    return image_aug

def run_aug():
    samples = len(glob2.glob('dataset_new/validation/' + emotion + '_filter/*')) + 1
    cnt = 0
    for s in range(1, samples):
        print('Loading sample', s, '/', samples)

        image = cv2.imread('dataset_new/validation/' + emotion + '_filter/(' + str(s) + ').jpg')
        if image is None:
            continue

        img_aug = generate(image)

        for img in img_aug:
            cv2.imwrite('dataset_new/validation/' + emotion + '/' + str(cnt) + '.jpg', img)
            cnt += 1

def run_filter():
    # Loading number of samples
    samples = len(glob2.glob('dataset_new/images/' + emotion + '/*')) + 1

    cnt = 0
    for s in range(1, samples):

        print('Loading sample', s, '/', samples)

        image = cv2.imread('dataset_new/images/' + emotion + '/' + str(s) + '.jpg')

        if image is None:
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        (height, width) = image.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104., 177., 123.))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                (x, y, w, h) = box.astype('int')
                gray_roi = gray[y:h, x:w]

                if gray_roi.shape[0] == 0 or gray_roi.shape[1] == 0:
                    continue
                else:
                    gray_roi = cv2.resize(gray_roi, (48, 48), interpolation=cv2.INTER_AREA)

                cv2.imwrite('dataset_new/validation/' + emotion + '_filter/' + str(cnt) + '.jpg', gray_roi)
                cnt += 1

# Emotion file
emotion = 'surprise'

# Uncomment this function if you need to pre-process input data
run_filter()

# Uncomment this function if you need to run augmentation data
# run_aug()


