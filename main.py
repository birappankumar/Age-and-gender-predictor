import cv2


def faceBox(faceNet,frame):
    # print(frame)
    frameHeight=frame.shape[0]
    frameWidth=frame.shape[1]
    blob=cv2.dnn.blobFromImage(frame, 1.0, (277,277), [104,117,123], swapRB=False)
    faceNet.setInput(blob)
    detection=faceNet.forward()
    bboxs=[]
    for i in range(detection.shape[2]):
        confidence=detection[0,0,i,2]
        if confidence>0.7:
            x1=int(detection[0,0,i,3]*frameWidth)
            y1=int(detection[0,0,i,4]*frameHeight)
            x2=int(detection[0,0,i,5]*frameWidth)
            y2=int(detection[0,0,i,6]*frameHeight)
            bboxs.append([x1,y1,x2,y2])
            #outer frame border
            border_thickness=30
            border_color=(255,0,0)
            x1_border=0
            y1_border=0
            x2_border=frame.shape[1]-1
            y2_border=frame.shape[0]-1
            #thickness of outer frame 
    
            cv2.rectangle(frame, (x1_border,y1_border),(x2_border,y2_border),border_color,border_thickness)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(135,206,235),1)
            #inner frame thickness
            inner_frame_thickness=10
            cv2.rectangle(frame,(x1, y1),(x2, y2),(0,191,255), inner_frame_thickness)
            
    return frame, bboxs
    # return detection


faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"

ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"

genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

faceNet=cv2.dnn.readNet(faceModel, faceProto)
ageNet=cv2.dnn.readNet(ageModel,ageProto)
genderNet=cv2.dnn.readNet(genderModel,genderProto)
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

#adding logo in frame
logo_path = "logo.png"  # Replace with your logo file path
logo = cv2.imread(logo_path)
logo = cv2.resize(logo, (200, 200))

#taking photo
video=cv2.VideoCapture(0)

padding=50
#giving position of logo
logo_x = 10
logo_y = 10
while True:
    ret,frame=video.read()
    frame,bboxs=faceBox(faceNet,frame)
    for bbox in bboxs:
        
        face = frame[max(0,bbox[1]-padding):min(bbox[3]+padding,frame.shape[0]-1),max(0,bbox[0]-padding):min(bbox[2]+padding, frame.shape[1]-1)]
        blob=cv2.dnn.blobFromImage(face,1.0,(227,227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPred=genderNet.forward()
        gender=genderList[genderPred[0].argmax()]


        ageNet.setInput(blob)
        agePred=ageNet.forward()
        age=ageList[agePred[0].argmax()]


        label="{},{}".format(gender,age)

        #adding logo of iit patna
        x_logo = 10  
        y_logo = 10  
        frame[y_logo:y_logo+logo.shape[0], x_logo:x_logo+logo.shape[1]] = cv2.addWeighted(frame[y_logo:y_logo+logo.shape[0], x_logo:x_logo+logo.shape[1]], 0.5, logo, 0.5, 0)

        #making text rectangle for giving output
        cv2.rectangle(frame,(bbox[0],bbox[1]-30),(bbox[2],bbox[1]),(0,191,255),-1) 
        cv2.putText(frame,label,(bbox[0], bbox[1]-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,255),2,cv2.LINE_AA)
        

    cv2.imshow("Age-Gender",frame)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
video.release()
cv2.destroyAllWindows()