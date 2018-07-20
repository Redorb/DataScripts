import imgaug as ia
from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
import os
import shutil
import zipfile

# TODO Make number of training images a cmd line argument

def loadImagesFromFolder(folder):
    images = []
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder,filename))
        images.append(img)
    return images
    
def generateAugSeq():
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)

    return iaa.Sequential(
        [
            sometimes(iaa.CropAndPad(
                percent=(-0.05, 0.1),
                pad_mode=ia.ALL,
                pad_cval=(0, 255)
            )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
                order=[0, 1],
                cval=(0, 255),
                mode=ia.ALL
            )),

            iaa.SomeOf((0, 5),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))),
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),
                        iaa.AverageBlur(k=(2, 7)),
                        iaa.MedianBlur(k=(3, 11)),
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),

                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5),
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5),
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    iaa.Invert(0.05, per_channel=True),
                    iaa.Add((-10, 10), per_channel=0.5),
                    iaa.AddToHueAndSaturation((-20, 20)),

                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.ContrastNormalization((0.5, 2.0))
                        )
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5),
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                ],
                random_order=True
            )
        ],
        random_order=True
    )

def mergeChannels(originalImage, rgbAugImage):
    try:
        return Image.merge("RGBA", (
            rgbAugImage.getchannel('R'),
            rgbAugImage.getchannel('G'),
            rgbAugImage.getchannel('B'),
            originalImage.getchannel('A')
        ))
    except ValueError:
        print("Error merging channels for file", originalImage.filename)
    
    return rgbAugImage 

def augmentImage(image):
    rgbImage = image.convert('RGB')
    images = [np.array(rgbImage)] * 1000 # Number of training images to create
    augseq = generateAugSeq()
    augImages = augseq.augment_images(images)
    images = []
    for augImage in augImages:
        images.append(Image.fromarray(augImage))
    return images

def createOrReplaceTrainingFolder():
    folder = 'training_data'
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder)

def deleteAndUnzipSprites():
    folder = "sprites"
    if os.path.exists(folder):
        shutil.rmtree(folder)
    zip_ref = zipfile.ZipFile("sprites.zip", 'r')
    zip_ref.extractall(folder)
    zip_ref.close()

def getSubfolders(folder):
    return [f.path for f in os.scandir(folder) if f.is_dir()]

def augmentImages(images):
    augImages = []
    for image in images:
        augImages.extend(augmentImage(image))
    return augImages


def writeImagesToFolder(images, folder):
    for idx, image in enumerate(images):
        image.save("{0}/{1}_{2}.png".format(folder, os.path.basename(folder), idx))

def createTrainingData():
    print("Unzipping sprites")
    deleteAndUnzipSprites()
    print("Structuring training folders")
    createOrReplaceTrainingFolder()
    spriteFolders = getSubfolders("sprites")
    itemFolders = []
    for itemGroupFolder in spriteFolders:
        itemFolders.extend(getSubfolders(itemGroupFolder))
    for itemFolder in itemFolders:
        print("Augmenting image {0}".format(os.path.basename(itemFolder)))
        trainingItemFolderPath = "training_data/{0}".format(os.path.basename(itemFolder))
        os.makedirs(trainingItemFolderPath)
        images = loadImagesFromFolder(itemFolder)
        writeImagesToFolder(augmentImages(images), trainingItemFolderPath)

if __name__ == "__main__":
    createTrainingData()
