import cv2
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
import numpy as np

inputFilePath = 'input.png'
inputImage = cv2.imread(inputFilePath, 0)

BIT_DEPTH = 8
NUM_TONES = float(2**BIT_DEPTH - 1)
FLOATING_POINT_REPRESENTATION = 'float32'
IMAGE_INTEGER_REPRESENTATION = 'uint8'

originalImage = inputImage.astype(FLOATING_POINT_REPRESENTATION)
normalizedImage = originalImage / NUM_TONES

pca = PCA(n_components=300).fit(normalizedImage)
mms = MinMaxScaler(feature_range=(0, NUM_TONES))


#COMPRESS
reducedImage = pca.transform(normalizedImage)
JPEG2000Encoding = cv2.imencode(".jpg", mms.fit_transform(reducedImage).astype(IMAGE_INTEGER_REPRESENTATION))[1]

#DECOMPRESS
JPEG2000Decoding = mms.inverse_transform(cv2.imdecode(JPEG2000Encoding, cv2.IMREAD_UNCHANGED).astype(FLOATING_POINT_REPRESENTATION))
reconstructedImage = pca.inverse_transform(JPEG2000Decoding) * NUM_TONES


mse = np.mean(np.power(originalImage - reconstructedImage, 2))
print mse