from flask import Flask,render_template,Response,request
import cv2
import numpy as np

app=Flask(__name__)

def generate_frames():
    camera=cv2.VideoCapture(0)
    while(True):
        success,frame=camera.read()
        if not success:
            break
        else:
            faceCascade = cv2.CascadeClassifier(r"E:\G2\Image Proccessing\Practicals\prj\Click the Image Smartly using Opencv\Click the Image Smartly using Opencv\dataset\haarcascade_frontalface_default.xml")
            smileCascade = cv2.CascadeClassifier(r"E:\G2\Image Proccessing\Practicals\prj\Click the Image Smartly using Opencv\Click the Image Smartly using Opencv\dataset\haarcascade_smile.xml")
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=faceCascade.detectMultiScale(gray,1.3,5)
            if len(faces)>0:
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                    roi_gray=gray[y:y+h,x:x+w]
                    roi_color=frame[y:y+h,x:x+w]
                    smiles=smileCascade.detectMultiScale(roi_gray,1.5,30,minSize=(50,50))
                    for i in smiles:
                        if len(i)>1:
                            cv2.putText(frame,"PERSON IS SMILING",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),3,cv2.LINE_AA)
                            path=r'E:\G2\Image Proccessing\Practicals\prj\Click the Image Smartly using Opencv\Click the Image Smartly using Opencv\static\images\sel.jpg'
                            cv2.imwrite(path,frame)
                            camera.release()
                            cv2.destroyAllWindows()
                            break

            ret,buffer=cv2.imencode(".jpg",frame)
            frame=buffer.tobytes()
        
        yield(b'--frame\r\n'b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')
            
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/smile.html')
def smile():
    return render_template('smile.html')

@app.route('/Video')
def Video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)
