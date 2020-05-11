from tensorflow.keras.models import load_model
from mtcnn.mtcnn import MTCNN
from os import listdir, makedirs
from os.path import join
from PIL import Image
from pyautogui import prompt
import cv2
import numpy as np
from pathlib import PurePath
from sklearn.svm import SVC
import pickle
from IPython.display import clear_output
np.set_printoptions(suppress=True)

detector = MTCNN()
embedding_size = 128

def l2_normalize(x, axis=-1, epsilon=1e-10):
    output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))
    return output

def loadImage(filepath):
    # load image from file
    image = Image.open(filepath)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    
    return pixels

def normalizeImage(image):
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    return image

def denormalizeImage(image):
    image = image - image.min()
    image = image / image.max()*255
    
    return image.astype(np.uint8)

def randomHSVShifts(image, shift_value=40):
    image = denormalizeImage(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int32)
    image[:, :, 0] += np.random.randint(-shift_value, shift_value)
    image[:, :, 1] += np.random.randint(-shift_value, shift_value)
    image[:, :, 2] += np.random.randint(-shift_value, shift_value)
    image = np.clip(image, 0, 255)
    image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2RGB)
    image = normalizeImage(image)
    return image
    
def extractFaces(image, normalize=True):
    results = detector.detect_faces(image)
    faces = np.zeros((0, 160, 160, 3), dtype=np.float32)
    coords = np.zeros((0, 4), dtype=np.int32)
    for i in range(len(results)):
        x1, y1, w, h = results[i]['box']
        x1 = abs(x1)
        y1 = abs(y1)
        w = abs(w)
        h = abs(h)
        face = image[y1:(y1+h), x1:(x1+w)]
        if face.shape[0]*face.shape[1]*face.shape[2] == 0:
            continue
        img = Image.fromarray(face)
        img = img.resize((160, 160))
        face = np.asarray(img)
        if normalize:
            face = normalizeImage(face)
        faces = np.append(faces, np.expand_dims(face, axis=0), axis=0)
        rect_coords = np.array([x1, y1, x1+w, y1+h]).reshape(1, 4)
        coords = np.append(coords, rect_coords, axis=0)
    return faces, coords

class Predictor:
    def __init__(self, model_path, face_database_path, classifier_path=None, identities_path=None, unbalanced=True, continuous_train=False, confidence_thresh_train=0.38, max_photos=20):
        self.unknown_thresh = 0.3
        self.N_aug = 5
        self.continuous_train = continuous_train
        self.confidence_thresh_train = confidence_thresh_train
        self.max_photos = max_photos
        print("Loading FaceNet...")
        self.model = load_model(model_path)
        print("Loading finished")
        self.face_database_path = face_database_path
        if identities_path is not None:
            self.identities = self.loadIdentities(identities_path)
        if classifier_path is not None:
            self.classifier = self.loadClassifier(classifier_path)
        else:
            print("Processing the database...")
            X, y = self.processDatabase(face_database_path)
            class_weights = self.computeClassWeights(y)
            print(class_weights)
            print("Processing finished")
            self.classifier = SVC(kernel='linear', probability=True, class_weight=class_weights)
            print("Training the classifier...")
            self.classifier.fit(X, y)
            print("Finished")
            self.X = X
            self.y = y
            self.saveIdentities("./identities_dict.pickle", self.identities)
            self.saveClassifier("./classifier.pickle", self.classifier)
            cv2.namedWindow('Detections')
            cv2.setMouseCallback('Detections', self.addNewIdentity)
    
    def processDatabase(self, face_database_path, detect_faces=True):
        folders = [join(face_database_path, f) for f in listdir(face_database_path)]
        names = []
        embeddings = np.zeros((0, embedding_size))
        for folder in folders:
            files = [join(folder, f) for f in listdir(folder)]
            for file in files:
                image = loadImage(file)
                if detect_faces:
                    faces, coords = extractFaces(image, normalize=False)
                    if faces.shape[0] == 0:
                        continue
                    image = faces[0]
                name = folder.split("\\", 1)[1]
                #folder_path = PurePath(folder)
                #name = str(folder_path.relative_to(folder_path.parts[0]))
                image = cv2.resize(image, (160, 160))
                image = normalizeImage(image)
                embedding = self.computeEmbeddings(image)
                embeddings = np.append(embeddings, embedding, axis=0)
                names.append(name)
        embeddings = l2_normalize(embeddings)
        labels, self.identities = self.toCategorical(names)
        return embeddings, labels
    
    def toCategorical(self, str_arr):
        values = list(set(str_arr))
        keys = list(range(len(values)))
        str_to_cat = dict(zip(values, keys))
        cat_to_str = dict(zip(keys, values))
        
        categorical = np.array([str_to_cat[string] for string in str_arr], dtype=np.int32)
        
        return categorical, cat_to_str
    
    def computeClassWeights(self, labels):
        unique = np.unique(labels)
        weights = {}
        for i in range(unique.shape[0]):
            label = unique[i]
            weight = 1 / labels[labels==label].shape[0]
            weights[label] = weight
        return weights
    
    def computeEmbeddings(self, images):
        return self.model.predict(images.reshape(-1, 160, 160, 3))
    
    def saveClassifier(self, path, classifier):
        with open(path, 'wb') as f:
            pickle.dump(classifier, f)
            
    def loadClassifier(self, path):
        with open(path, 'rb') as f:
            classifier = pickle.load(f)
        return classifier
    
    def saveIdentities(self, path, identities):
        with open(path, 'wb') as f:
            pickle.dump(identities, f)            
        
    def loadIdentities(self, path):
        with open(path, 'rb') as f:
            identities = pickle.load(f)
        return dict(identities)
    
    def saveConfig(self, clf_path, id_path):
        self.saveClassifier(clf_path, self.classifier)
        self.saveIdentities(id_path, self.identities)
        
    def predictFaces(self, image):
        faces, coords = extractFaces(image)
        if faces.shape[0] == 0:
            return image
        emb = self.computeEmbeddings(faces)
        emb = l2_normalize(emb)
        clear_output(wait=True)
        classes = self.classifier.predict(emb)
        probs = self.classifier.predict_proba(emb)
        self.coords = coords
        self.faces = faces
        for i in range(classes.shape[0]):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.rectangle(image, (coords[i, 0], coords[i, 1]), (coords[i, 2], coords[i, 3]), (0, 255, 0), 2)
            clear_output(wait=True)
            if np.max(probs[i]) < self.unknown_thresh:
                label = "Unknown person"
            else:
                label = self.identities[classes[i]]
            # Continous train
            if np.max(probs[i])  > self.confidence_thresh_train and self.continuous_train:
                self.continuousTrain(emb[i], classes[i])
            cv2.putText(image, label, (coords[i, 0], coords[i, 1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Detections", image)  
        
    def inRectangle(self, x, y):
        rect_idx = None
        if self.coords is None:
            return rect_idx
        for i in range(self.coords.shape[0]):
            x1, y1, x2, y2 = self.coords[i]
            if x > x1 and x < x2 and y > y1 and y < y2:
                rect_idx = i
                return rect_idx
    
    def addNewIdentity(self, event, x, y, flags, param):
        rect_idx = self.inRectangle(x, y)
        if event == cv2.EVENT_LBUTTONDBLCLK and rect_idx is not None:
            name = prompt(text='Enter the name of this person', title='Add new face image' , default='')
            if name is None:
                return
            if name in self.identities.values():
                identities_to_number = dict([(value, key) for key, value in self.identities.items()]) 
                class_number = identities_to_number[name]
                new_identity = False
            else:
                class_number = np.max(self.y) + 1
                makedirs(self.face_database_path+"/"+name)
                self.identities.update({class_number: name})
                new_identity = True
            if new_identity:
                for i in range(self.N_aug):
                    face_img = self.faces[rect_idx] 
                    face_img = randomHSVShifts(face_img)
                    emb = l2_normalize(self.computeEmbeddings(face_img))
                    # Appending embeddings and labels
                    self.X = np.append(self.X, emb, axis=0)
                    self.y = np.append(self.y, class_number.reshape(1,), axis=0)
                    # Saving the image to the database
                    tosave = cv2.cvtColor(denormalizeImage(face_img), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(self.face_database_path+"/"+name+"/"+name+str(self.y[self.y==class_number].shape[0])+".jpg", tosave)
            else:
                face_img = self.faces[rect_idx]
                emb = l2_normalize(self.computeEmbeddings(face_img))
                # Appending embeddings and labels
                self.X = np.append(self.X, emb, axis=0)
                self.y = np.append(self.y, class_number.reshape(1,), axis=0)
                # Saving the image to the database
                tosave = cv2.cvtColor(denormalizeImage(face_img), cv2.COLOR_RGB2BGR)
                cv2.imwrite(self.face_database_path+"/"+name+"/"+name+str(self.y[self.y==class_number].shape[0])+".jpg", tosave)
            # Retraining the classifier
            class_weights = self.computeClassWeights(self.y)
            self.classifier = SVC(kernel='linear', probability=True, class_weight=class_weights)
            self.classifier.fit(self.X, self.y)

    def continuousTrain(self, emb, label):
        if self.y[self.y==label].shape[0] >= self.max_photos:
            return 
        print("Adding %s embedding to train data"%self.identities[label])
        self.X = np.append(self.X, emb.reshape(1, -1), axis=0)
        self.y = np.append(self.y, np.array([label]), axis=0)
        # Retraining the classifier
        class_weights = self.computeClassWeights(self.y)
        self.classifier = SVC(kernel='linear', probability=True, class_weight=class_weights)
        self.classifier.fit(self.X, self.y)