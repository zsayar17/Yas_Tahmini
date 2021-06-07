
import numpy as np
import cv2

import keras
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dropout, Activation
from keras.preprocessing import image





def VggYuzModel():
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))



	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))




	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))




	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))




	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, (3, 3), activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))




	model.add(Convolution2D(4096, (7, 7), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(4096, (1, 1), activation='relu'))
	model.add(Dropout(0.5))
	model.add(Convolution2D(2622, (1, 1)))
	model.add(Flatten())
	
	
    
    
	return model



def YasModel():
    
    
    
    
    
	model = VggYuzModel()
	
	model_output = Sequential()
	model_output = Convolution2D(101, (1, 1), name='predictions')(model.layers[-4].output)
	model_output = Flatten()(model_output)
	model_output = Activation('softmax')(model_output)
	
	yas_model = Model(inputs=model.input, outputs=model_output)
	
	yas_model.load_weights("age_model_weights.h5")

 

	return yas_model




	
yas_model = YasModel()




#çıkış indeksleri aldık
cikis_index = np.array([i for i in range(0, 101)])


#yüzümüzün bulunduğu aralıkları aldık
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#webcami açtık
cap = cv2.VideoCapture(0) 



while(True):
    
    
	ret, img = cap.read()
   
	yuzler = face_cascade.detectMultiScale(img, 1.3, 5)
    
	for (x,y,w,h) in yuzler:
		if w > 100: #küçük yüzleri yoksadyık
		

			#görüntü üzerinde dikdörtgen çiziyoruz

			cv2.rectangle(img,(x,y),(x+w,y+h),(0,150,200),3) #karenin boyutlarını veriyor
        


			#tesip edilen yüzü alıyoruz
			Yuz = img[int(y):int(y+h), int(x):int(x+w)] 
			

			
			try:
				#vgg face de predict edebilmek için 224,224 olarak resize ettik
				Yuz = cv2.resize(Yuz, (224, 224))
				
				img_pixels = image.img_to_array(Yuz)
				img_pixels = np.expand_dims(img_pixels, axis = 0)
				img_pixels /= 255
				
				#yası tahmin ettik 
				yas_tahmin = yas_model.predict(img_pixels)
				belirli_yas = str(int(np.floor(np.sum(yas_tahmin * cikis_index, axis = 1))[0]))
				

			
				
				#yas için labelı koyduk
				cv2.putText(img, belirli_yas, (x+int(w/2), y - 45), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 111, 255), 2)
				
			
            
            
            
			except Exception as e:
				print("exception",str(e))
			
            
            
                    
	cv2.imshow('img',img)
	
	if cv2.waitKey(1) & 0xFF == ord('q'): #çıkış yapmak için q bas
		break
	
    
    
    
#capture temizle		
cap.release()
cv2.destroyAllWindows()
