from moviepy.editor import *

videoclip1 = VideoFileClip("./House Tour_ An eclectic black and white bungalow in Singapore.mp4")
audioclip = videoclip1.audio

videoclip2 = VideoFileClip("./House Tour.avi")
speed = videoclip2.duration/videoclip1.duration
print(speed)
sped_up_video = videoclip2.speedx(factor=speed)
sped_up_video.audio = audioclip
sped_up_video.write_videofile("./House Tour(YOLO).mp4")
