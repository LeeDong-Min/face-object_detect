from moviepy.editor import *

videoclip1 = VideoFileClip("./video/키싱부스2-공식예고편.mp4")
audioclip = videoclip1.audio

videoclip2 = VideoFileClip("./video/output_hog3.mp4")
speed = videoclip2.duration/videoclip1.duration
print(speed)
sped_up_video = videoclip2.speedx(factor=speed)
sped_up_video.audio = audioclip
sped_up_video.write_videofile("./video/[박주연]키싱부스2_hog.mp4")
