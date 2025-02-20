import cv2
import numpy as np
import os

from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

spectrograms_path = "./pictures_with_shift_crop"
spectrogram_files = [os.path.join(spectrograms_path, f) for f in os.listdir(spectrograms_path) if f.endswith('.png')]
labels = []
sift = cv2.SIFT_create()
for person_id in range(6):
    for i in range(10):
        labels.append(person_id) 
descriptors_list = []

for file in spectrogram_files:
    img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
    
    keypoints, descriptors = sift.detectAndCompute(img, None)
    
    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow("Keypoints", img_with_keypoints)
    # cv2.waitKey(0)
    
    if descriptors is not None:
        descriptors_list.append(descriptors)

cv2.destroyAllWindows()
all_descriptors = np.vstack(descriptors_list)

num_clusters = 30
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(all_descriptors)


spectrogram_features = []

for descriptors in descriptors_list:
    labels_clusters = kmeans.predict(descriptors)
    
    histogram, _ = np.histogram(labels_clusters, bins=range(num_clusters + 1), density=True)
    spectrogram_features.append(histogram)

spectrogram_features = np.array(spectrogram_features)
labels = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(spectrogram_features, labels, test_size=0.3, random_state=42)
print(y_train, y_test)
svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(spectrogram_features, labels)

y_pred = svm.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, zero_division=0))

