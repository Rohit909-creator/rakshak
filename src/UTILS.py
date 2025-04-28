import cv2
import os

os.makedirs("./frames", exist_ok=True)


def video2frames(path = "Dubai road accident CCTV Footage Revealed By Dubai Police.mp4"):
    
    cam = cv2.VideoCapture(path)
    i = 0
    while True:
        
        ret, frame = cam.read()
        
        if not ret:
           break
       
        cv2.imwrite(f"./frames/frame{i}.jpg", frame)
        i+=1

if __name__ == "__main__":
    inp = input("Wanna proceed making video2frame (Y/N):")
    if inp.lower() == "y":
        video2frames()
    else:
        print("Ok cool Dude")