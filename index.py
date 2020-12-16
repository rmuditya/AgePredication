import cv2
import imutils
import time
import numpy as np

cap = cv2.VideoCapture(0)

cap.set(3, 480) #set width
cap.set(4, 640) #set height

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0, 2)', '(4, 6)', '(8, 12)', '(15, 20)', '(25, 32)', '(38, 43)', '(48, 53)', '(60, 100)']
gender_list = ['Male', 'Female']

def initialize_caffe_models():
	
	age_net = cv2.dnn.readNetFromCaffe(
		'data/deploy_age.prototxt', 
		'data/age_net.caffemodel')

	gender_net = cv2.dnn.readNetFromCaffe(
		'data/deploy_gender.prototxt', 
		'data/gender_net.caffemodel')

	return(age_net, gender_net)

def read_from_camera(age_net, gender_net):
	font = cv2.FONT_HERSHEY_PLAIN

	while True:

		ret, image = cap.read()

		face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		faces = face_cascade.detectMultiScale(gray, 1.1, 5)

		if(len(faces)>0):
			print("Found {} faces".format(str(len(faces))))

		for (x, y, w, h )in faces:
			startX,startY = x,y
			endX = x+w
			endY = y+h
				# starting left
			cv2.line(image, (startX, startY), (startX+20, startY), (0,0,0), thickness=2)
			cv2.line(image, (startX, startY), (startX, startY+20), (0,0,0), thickness=2)
			# ending right
			cv2.line(image, (endX, endY), (endX, endY-20), (0,0,0), thickness=2)
			cv2.line(image, (endX, endY), (endX-20, endY), (0,0,0), thickness=2)

			# ending left
			cv2.line(image,(startX+20, endY),(startX, endY), (0,0,0), thickness=2)
			cv2.line(image,(startX, endY-20),(startX, endY), (0,0,0), thickness=2)
			
			# starting left
			cv2.line(image,(endX, startY+20),(endX, startY), (0,0,0), thickness=2)
			cv2.line(image,(endX-20, startY),(endX, startY), (0,0,0), thickness=2)

			# cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 0), 2)

			# Get Face 
			face_img = image[y:y+h, h:h+w].copy()
			blob = cv2.dnn.blobFromImage(face_img, 1, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

			#Predict Gender
			gender_net.setInput(blob)
			gender_preds = gender_net.forward()
			gender = gender_list[gender_preds[0].argmax()]
			print("Gender Predicted: " + gender)

			#Predict Age
			age_net.setInput(blob)
			age_preds = age_net.forward()
			age = age_list[age_preds[0].argmax()]
			print("Age Predicted: " + age)

			# overlay_text = "%s %s" % (gender, age)
			# cv2.putText(image, overlay_text, (x+200, y+10), font, 1.2, (82,99,100), 2, cv2.LINE_AA)
			cv2.putText(image, "Gender: " + str(gender), (x+200, y+10), font, 1.2, (82,99,100), 2, cv2.LINE_AA)
			cv2.putText(image, "Age: " + str(age), (x+200, y+30), font, 1.2, (82,99,100), 2, cv2.LINE_AA)


		cv2.imshow('Predictation', image)

		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

if __name__ == "__main__":
	age_net, gender_net = initialize_caffe_models()

	read_from_camera(age_net, gender_net)




