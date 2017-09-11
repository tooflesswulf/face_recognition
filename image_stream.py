# import matplotlib.pyplot as plt
import cv2
import match_people
import face_recognition

# this would be the image stream from camera.
im = cv2.imread('ashwin2.jpg')

im = cv2.resize(im, (0,0), fx=.25, fy=.25)

print(im.shape)


while True:
    face_locations = face_recognition.face_locations(im, number_of_times_to_upsample=0, model="cnn")

    print("I found {} face(s) in this photograph.".format(len(face_locations)))

    to_show = im.copy()
    for face_location in face_locations:
        # Print the location of each face in this image
        top, right, bottom, left = face_location
        print(face_location)

        cv2.rectangle(to_show, (right, top), (left, bottom), (255,0,0), 3)

    cv2.imshow('ashwins face', to_show)
    k = cv2.waitKey(1) & 0xFF

    if k==ord('q'):
        break



print('Done')
