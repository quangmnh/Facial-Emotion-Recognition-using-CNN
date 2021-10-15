from imgaug import augmenters as iaa
import cv2
import glob2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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

samples = len(glob2.glob('dataset_new/images/sad/*')) + 1

cnt = 0
for i in range(1, samples):
    image = cv2.imread('dataset_new/images/sad/' + str(i) + '.jpg')
    if image is None: continue
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if faces is not None:
        for x, y, w, h in faces:
            cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), (0, 255, 0), 2)
            gray_roi = gray[y:y+h, x:x+w]
            gray_roi = cv2.resize(gray_roi, (48, 48), interpolation=cv2.INTER_AREA)

            img_aug = generate(gray_roi)

            for img in img_aug:
                # cv2.imshow('test', img)
                cv2.imwrite('dataset_new/validation/sad/' + str(cnt) + '.jpg', img)
                cnt += 1
                # cv2.waitKey(0)

    else:
        print('No face')

# for i in range(0, samples):
#     image = cv2.imread('dataset_new/images/happy_gen/' + str(i) + '.png')
#     img_aug = generate(image)
#
#     for img in img_aug:
#         cv2.imwrite('dataset_new/validation/happy/' + str(cnt) + '.png', img)
#         cnt += 1
